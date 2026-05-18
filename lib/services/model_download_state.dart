import 'inference_service.dart' show CancelToken;

/// Per-model download lifecycle. Used to drive the Models tab UI, the
/// onboarding download screen, and the floating download banner from a
/// single source of truth.
///
/// Transitions:
///   idle → downloading        (user taps Download)
///   downloading → installed   (Future completes normally)
///   downloading → idle        (user taps Cancel — partial file discarded)
///   downloading → failed      (network/disk error)
///   any         → idle        (user taps Delete on an installed model)
///
/// Note: There is intentionally NO `paused` state. `flutter_gemma`'s
/// underlying SmartDownloader disables pause/resume for HuggingFace URLs
/// because HF's weak ETags break the resume machinery — pause would
/// silently restart from zero, which is misleading. Cancel-only keeps
/// the UI honest.
enum DownloadState {
  /// Not on disk and not currently downloading. UI shows a Download button.
  idle,

  /// Active network fetch. UI shows the progress bar + Cancel button.
  downloading,

  /// Fully downloaded — file is on disk and registered with the SDK.
  installed,

  /// Last attempt errored out (network, disk full, signing). UI shows
  /// the message and a Retry button.
  failed,
}

/// Snapshot of one model's download state. Held in a Map keyed by
/// `LlmModel` on `_AshAppState`. Immutable — `_updateDownload` produces
/// a fresh copy on every transition so `setState` sees a real change.
class ModelDownloadInfo {
  const ModelDownloadInfo({
    this.state = DownloadState.idle,
    this.progress = 0,
    this.etaMin = 0,
    this.cancelToken,
    this.errorMessage,
  });

  /// Where this model is in its lifecycle. See [DownloadState] for the
  /// transition diagram.
  final DownloadState state;

  /// 0..1 fraction completed. Meaningful in [DownloadState.downloading]
  /// and [DownloadState.paused] (where it freezes at the cancel point).
  /// 0 otherwise.
  final double progress;

  /// Cached ETA in minutes. Computed by the host from the model's listed
  /// total size and the latest progress. Shown next to the bar.
  final int etaMin;

  /// Token whose `.cancel(reason)` interrupts the in-flight install
  /// Future. Only non-null in [DownloadState.downloading]. The host
  /// clears it when transitioning out of `downloading`.
  final CancelToken? cancelToken;

  /// Human-readable failure detail. Only meaningful in
  /// [DownloadState.failed].
  final String? errorMessage;

  ModelDownloadInfo copyWith({
    DownloadState? state,
    double? progress,
    int? etaMin,
    CancelToken? cancelToken,
    String? errorMessage,
    bool clearCancelToken = false,
    bool clearError = false,
  }) {
    return ModelDownloadInfo(
      state: state ?? this.state,
      progress: progress ?? this.progress,
      etaMin: etaMin ?? this.etaMin,
      cancelToken: clearCancelToken ? null : cancelToken ?? this.cancelToken,
      errorMessage: clearError ? null : errorMessage ?? this.errorMessage,
    );
  }

  bool get isActiveDownload => state == DownloadState.downloading;
  bool get isInstalled => state == DownloadState.installed;
}

/// Live ETA estimator for a single in-flight download.
///
/// The old algorithm was just `baselineMinutes * (1 - progress)` — a linear
/// walk-down from a hardcoded baseline. That produced numbers that looked
/// plausible but didn't actually track: users on fast WiFi watched it tick
/// down faster than the bar filled, and users on slow connections watched
/// it hit zero with the bar still halfway full.
///
/// This estimator only reports a number once it has seen real measured
/// progress. Until then, [update] returns 0 — the UI renders that as a
/// quiet "Calculating…" rather than inventing a misleading baseline.
/// Once there is a meaningful sample, an elapsed-based projection
/// (`elapsed / progress` extrapolated to 100%) kicks in immediately;
/// after the first couple of seconds, an EMA over observed
/// progress-per-second takes over so the number reacts to actual
/// bandwidth shifts.
///
/// One estimator instance per active download; lives in
/// `_AshAppState._etaEstimators` and is dropped on
/// completion / cancel / fail.
class EtaEstimator {
  EtaEstimator() : _start = DateTime.now() {
    _lastTime = _start;
  }

  /// EMA smoothing factor in [0, 1]. Higher = more reactive to recent
  /// speed; lower = more stable but slower to react to bandwidth changes
  /// (e.g. moving between WiFi cells). 0.3 keeps the number from
  /// jittering on every progress tick while still adapting within a few
  /// seconds when speed shifts.
  static const _alpha = 0.3;

  /// Minimum elapsed seconds before we trust the smoothed speed. Until
  /// then we use the simple elapsed-based projection, which converges
  /// faster than EMA from a cold start.
  static const _warmupSec = 3;

  /// Minimum progress fraction before we attempt any projection at all.
  /// Below this, the numerator is too noisy (one CDN handshake spike
  /// can produce a wildly wrong "X hours left"). At 0.005 (0.5%) we've
  /// seen a meaningful amount of data; below it we just say
  /// "Calculating…".
  static const _minProgressForEstimate = 0.005;

  final DateTime _start;
  late DateTime _lastTime;
  double _lastProgress = 0;

  /// Progress fraction per second, EMA-smoothed. 0 until the first
  /// real positive sample arrives.
  double _smoothedFracPerSec = 0;

  /// Feed a fresh progress reading. Returns the current best estimate
  /// in minutes (ceil), or 0 to signal "no estimate yet" (cold start or
  /// download just kicked off — the UI renders this as
  /// "Calculating…"). 0 is also returned when the download is complete,
  /// which is unambiguous from context.
  int update(double progress) {
    if (progress >= 1.0) return 0;

    final now = DateTime.now();
    final dt = now.difference(_lastTime).inMilliseconds / 1000.0;
    final dp = progress - _lastProgress;

    // Update the smoothed speed only on forward progress. dt may be 0
    // when two updates arrive in the same millisecond — skip those so
    // we don't divide by zero.
    if (dt > 0 && dp > 0) {
      final instant = dp / dt;
      _smoothedFracPerSec = _smoothedFracPerSec <= 0
          ? instant
          : _alpha * instant + (1 - _alpha) * _smoothedFracPerSec;
      _lastProgress = progress;
      _lastTime = now;
    }

    // No real signal yet → tell the caller we don't know.
    if (progress < _minProgressForEstimate) return 0;

    final elapsedSec = now.difference(_start).inSeconds;

    // Cold-start branch: lean on the simple "elapsed/progress" projection
    // until the EMA has had a few seconds to settle. This is noisier per
    // tick but converges to a believable number within ~1 second of real
    // download activity.
    if (_smoothedFracPerSec <= 0 || elapsedSec < _warmupSec) {
      if (elapsedSec <= 0) return 0;
      final projectedTotalSec = elapsedSec / progress;
      final remainSec = (projectedTotalSec - elapsedSec).clamp(0, 1 << 30);
      return _secondsToMinutes(remainSec.toDouble());
    }

    final remainSec = (1.0 - progress) / _smoothedFracPerSec;
    return _secondsToMinutes(remainSec);
  }

  /// Round up so "59 seconds left" reads as "1 min" instead of "0 min"
  /// (which would look done while the bar is still filling).
  static int _secondsToMinutes(double seconds) {
    if (seconds <= 0) return 0;
    return (seconds / 60.0).ceil();
  }
}
