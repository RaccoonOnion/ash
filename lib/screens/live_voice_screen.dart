import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';
import 'package:image_picker/image_picker.dart';

import '../models/chat.dart';
import '../services/context_estimator.dart';
import '../services/inference_service.dart';
import '../services/tts_sanitizer.dart';
import '../services/voice_service.dart';
import '../theme/ash_theme.dart';

/// Full-screen, hands-free conversation:
///   listen → transcribe → query model → speak the reply → listen again
///
/// Each completed turn is appended to the underlying [chat] via callbacks so
/// the regular chat surface shows the same conversation when the user exits.
///
/// Visual design follows ChatGPT's voice mode: a big animated orb dominates
/// the screen, default UI is intentionally bare, and a captions sheet
/// slides up only when the user asks for it.
class LiveVoiceScreen extends StatefulWidget {
  const LiveVoiceScreen({
    required this.chat,
    required this.service,
    required this.voiceService,
    required this.onUserTurn,
    required this.onAssistantTurn,
    required this.priorHistorySnapshot,
    required this.onExit,
    required this.onContextExtend,
    required this.onContextTrim,
    super.key,
  });

  final Chat chat;
  final InferenceService service;
  final VoiceService voiceService;
  final ValueChanged<String> onUserTurn;

  /// Fires once per completed assistant turn in live mode. Receives the
  /// final transcript plus any RAG sources that grounded the reply. The
  /// sources list is what `query()` yields on its preflight chunk —
  /// dropping it here is how live-mode messages used to lose their
  /// citation chips when the user came back to text chat.
  final void Function(String text, List<MessageSource>? sources)
      onAssistantTurn;
  final List<HistoryTurn> Function() priorHistorySnapshot;
  final VoidCallback onExit;
  /// Bump maxTokens by 2048 (capped at 32k) and reload the engine.
  /// Awaited by [_autoManageContext] before each user turn so the model
  /// has room for the next reply — the engine reload is folded into the
  /// existing "thinking" pause rather than surfacing as a UI banner.
  final Future<int> Function() onContextExtend;
  /// Drop oldest ~30% of messages + reset the inference chat session.
  /// Fallback when [onContextExtend] returns the same value (already at
  /// the 32k cap). Cheaper than extend (no engine reload).
  final Future<int> Function() onContextTrim;

  @override
  State<LiveVoiceScreen> createState() => _LiveVoiceScreenState();
}

enum _LiveStatus { initializing, listening, thinking, speaking, error }

class _LiveVoiceScreenState extends State<LiveVoiceScreen>
    with TickerProviderStateMixin {
  _LiveStatus _status = _LiveStatus.initializing;
  String _userPartial = '';
  String _currentAssistantTurn = '';
  /// Sources for the current assistant turn — captured from the
  /// preflight chunk that `query()` emits before any tokens. Stays null
  /// until that chunk lands (and stays null entirely on RAG-off / image
  /// queries). Saved alongside the final transcript so the chip row
  /// renders when the user returns to the text chat surface.
  List<MessageSource>? _currentAssistantSources;

  /// Raw model tokens not yet flushed to TTS. We accumulate here until
  /// we have at least one complete sentence (or a newline), then run
  /// [sanitizeForTts] over that whole segment and forward the cleaned
  /// version to the voice service. Sanitizing per-chunk is broken for
  /// list markers — Gemma streams `"1"`, `"."`, `" Apply"` as three
  /// tokens, and the line-anchored regex `^\s*\d+\.\s+` never sees a
  /// complete `"1. "` to strip. By the time the voice service
  /// re-assembles those into a sentence, the marker is already in. So
  /// we move the sentence-boundary slice BEFORE the sanitize step
  /// here. Voice service still slices again on its side — cheap, and
  /// the second pass finds nothing new to do.
  String _ttsRawBuffer = '';

  /// Slice on sentence-end punctuation followed by whitespace, OR on
  /// a run of newlines. The trailing whitespace is INCLUDED in the
  /// match so the segment ends at the boundary instead of just before
  /// it — that matters for the markdown-strip regex, which needs to
  /// see "1. " (digit, period, space) as one piece. Slicing at just
  /// `.` would leave the trailing space in the next segment and the
  /// numbered-list strip wouldn't fire on the line that starts there.
  static final _ttsBoundaryRegex = RegExp(r'[.!?]\s+|\n+');
  String? _errorMessage;
  bool _exiting = false;
  bool _isFirstTurn = true;
  bool _showCaptions = false;
  bool _awaitingPostReplyResume = false;

  StreamSubscription<String>? _partialSub;
  StreamSubscription<String>? _finalSub;
  StreamSubscription<bool>? _speakingSub;
  StreamSubscription<String>? _errorSub;
  StreamSubscription<InferenceChunk>? _querySub;
  // Safety-net timer: if TTS's speaking stream silently fails to fire
  // (iOS audio session interleaving edge case), force-resume the mic
  // after a few seconds so the user isn't stranded in mid-conversation.
  Timer? _resumeWatchdog;
  // STT silence watchdog. iOS's SFSpeechRecognizer auto-stops on
  // prolonged silence but doesn't always fire a `finalStream` event —
  // when that happens the app sits in `listening` state forever (the
  // iPhone yellow mic indicator goes dark). This timer re-arms STT
  // periodically while we're supposed to be listening and seeing no
  // activity. The heartbeat is reset on every partial transcript.
  Timer? _sttWatchdog;
  DateTime? _lastSttActivity;
  // Window of no STT partials before we assume the recognizer is dead.
  // Must be longer than voiceService.listeningPatience (which drives
  // pauseFor) plus enough buffer that a legitimately-quiet user isn't
  // restarted mid-thought; short enough that a truly stuck mic doesn't
  // strand the user for too long. Computed dynamically so it tracks the
  // Settings → Listening patience slider.
  Duration get _sttSilenceWindow =>
      widget.voiceService.listeningPatience + const Duration(seconds: 7);

  // Slow pulse: the orb's "breathing" between state-driven peaks.
  late final AnimationController _slowPulse = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 2600),
  )..repeat(reverse: true);

  // Faster pulse used during active states (speaking / listening) to give
  // the orb a more alive feel.
  late final AnimationController _fastPulse = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 800),
  )..repeat(reverse: true);

  // (Removed: dedicated spin controller for the thinking ring. We now
  // use a CircularProgressIndicator which carries its own ticker — one
  // less animation controller to dispose and one less listener for the
  // orb's AnimatedBuilder to rebuild on.)

  @override
  void initState() {
    super.initState();
    _start();
  }

  @override
  void dispose() {
    _slowPulse.dispose();
    _fastPulse.dispose();
    super.dispose();
  }

  // ──────────────────────────────────────────────────────────────
  // State machine — unchanged from prior version
  // ──────────────────────────────────────────────────────────────

  Future<void> _start() async {
    final status = await widget.voiceService.getStatus();
    if (!status.sttReady || !status.micPermitted) {
      if (!mounted) return;
      setState(() {
        _status = _LiveStatus.error;
        _errorMessage =
            'Voice needs two iOS permissions. Open Settings → Privacy & '
            'Security and enable BOTH:\n'
            '• Microphone → Ash\n'
            '• Speech Recognition → Ash\n'
            'Then come back and tap to retry.';
      });
      return;
    }
    // _start can be called more than once (error-recovery path in
    // _retryFromError). Cancel any existing subscriptions before
    // re-subscribing — without this, every callback fires N times
    // after the Nth recovery and the audio flow gets racy.
    await _partialSub?.cancel();
    await _finalSub?.cancel();
    await _speakingSub?.cancel();
    await _errorSub?.cancel();
    _partialSub = widget.voiceService.partialStream.listen(_onPartial);
    _finalSub = widget.voiceService.finalStream.listen(_onFinal);
    _speakingSub =
        widget.voiceService.speakingStream.listen(_onSpeakingChanged);
    _errorSub = widget.voiceService.errorStream.listen(_onVoiceError);
    await _beginUserTurn();
  }

  void _onVoiceError(String msg) {
    if (_exiting || !mounted) return;
    final lower = msg.toLowerCase();
    if (lower.contains('no_match') ||
        lower.contains('no_speech') ||
        lower.contains('speech_timeout')) {
      _beginUserTurn();
      return;
    }
    setState(() {
      _status = _LiveStatus.error;
      _errorMessage = 'Voice error: $msg';
    });
  }

  Future<void> _beginUserTurn() async {
    if (_exiting) return;
    setState(() {
      _status = _LiveStatus.listening;
      _userPartial = '';
    });
    try {
      if (widget.voiceService.isRecording) {
        await widget.voiceService.cancelRecording();
      }
      await widget.voiceService.startRecording();
      _armSttWatchdog();
    } catch (e) {
      debugPrint('[live-voice] startRecording failed: $e');
      if (mounted) {
        setState(() {
          _status = _LiveStatus.error;
          _errorMessage = 'Couldn’t start the mic: $e';
        });
      }
    }
  }

  /// (Re-)arm the STT silence watchdog. Fires periodically while we
  /// expect to be listening; if the last partial transcript is older
  /// than [_sttSilenceWindow], we force-restart STT. Without this the
  /// iOS yellow mic indicator silently disappears after an OS-level
  /// auto-stop and the user has to manually toggle mute on/off.
  void _armSttWatchdog() {
    _sttWatchdog?.cancel();
    _lastSttActivity = DateTime.now();
    _sttWatchdog =
        Timer.periodic(const Duration(seconds: 4), (_) {
      if (_exiting) return;
      if (_status != _LiveStatus.listening) return;
      // If the service thinks it's still listening AND we got an
      // activity ping recently, do nothing — STT is healthy.
      final last = _lastSttActivity;
      if (last != null &&
          DateTime.now().difference(last) < _sttSilenceWindow) {
        return;
      }
      // No partials for [_sttSilenceWindow]. iOS may have auto-stopped
      // the recognizer at the OS level without firing finalStream. Force
      // a fresh listen() — _beginUserTurn re-arms this watchdog too.
      debugPrint('[live-voice] STT watchdog: no activity for '
          '${_sttSilenceWindow.inSeconds}s — re-arming mic');
      _beginUserTurn();
    });
  }

  void _onPartial(String partial) {
    if (_exiting) return;
    // Watchdog uses this as the freshness signal — every partial counts
    // as STT being healthy, even an empty one (means the recognizer is
    // alive and listening even if no speech yet).
    _lastSttActivity = DateTime.now();
    setState(() => _userPartial = partial);
  }

  Future<void> _onFinal(String text) async {
    if (_exiting) return;
    final clean = text.trim();
    if (clean.isEmpty) {
      await _beginUserTurn();
      return;
    }
    _sttWatchdog?.cancel();
    setState(() {
      _userPartial = clean;
      _status = _LiveStatus.thinking;
      _currentAssistantTurn = '';
      _currentAssistantSources = null;
      _ttsRawBuffer = '';
    });
    widget.onUserTurn(clean);

    // Auto-manage context BEFORE we hit the engine. In text chat the
    // user sees the warning banner and decides; in live mode there's no
    // banner to read, so we silently extend (or trim if already at the
    // cap) folded into the existing "thinking" pause.
    await _autoManageContextForLive(clean);
    if (_exiting) return;

    final priorHistory =
        _isFirstTurn ? widget.priorHistorySnapshot() : const <HistoryTurn>[];
    _isFirstTurn = false;

    try {
      _querySub = widget.service
          .query(
            chatId: widget.chat.id,
            prompt: clean,
            priorHistory: priorHistory,
            settings: widget.chat.inferenceSettings,
            // Voice-tuned system prompt: short prose, no markdown, no
            // emoji. The TTS layer also strips any stray formatting from
            // each chunk before speaking, but having the model not emit
            // it in the first place produces a more natural reply.
            liveMode: true,
          )
          .listen(_onChunk, onDone: _onQueryDone, onError: _onQueryError);
    } catch (e) {
      debugPrint('[live-voice] query setup failed: $e');
      if (mounted) {
        setState(() {
          _status = _LiveStatus.error;
          _errorMessage = 'Inference failed to start.';
        });
      }
    }
  }

  void _onChunk(InferenceChunk chunk) {
    if (_exiting) return;
    // Preflight chunk: carries the RAG sources that grounded this reply.
    // Arrives before any token text, so capture and bail before the
    // text-accumulation path runs.
    if (chunk.sources != null) {
      _currentAssistantSources = chunk.sources;
      return;
    }
    if (chunk.isDone) return;
    _currentAssistantTurn += chunk.text;
    setState(() => _status = _LiveStatus.speaking);
    // TTS path: accumulate into the raw buffer, then drain whole
    // sentences (or newlines) at a time. Sanitizing at the sentence
    // level lets the line-anchored regex strip list markers like
    // "1. " or "- " that Gemma's tokenizer would otherwise stream
    // across multiple chunks — the per-chunk regex couldn't see the
    // full marker and the user would hear "one period apply…". The
    // captions surface still shows raw markdown via
    // `_currentAssistantTurn`; only TTS gets the cleaned text.
    _ttsRawBuffer += chunk.text;
    while (true) {
      final match = _ttsBoundaryRegex.firstMatch(_ttsRawBuffer);
      if (match == null) break;
      final end = match.end;
      final segment = _ttsRawBuffer.substring(0, end);
      _ttsRawBuffer = _ttsRawBuffer.substring(end);
      final spoken = sanitizeForTts(segment, trimEnds: false);
      if (spoken.isNotEmpty) {
        widget.voiceService.feedTtsChunk(spoken);
      }
    }
  }

  Future<void> _onQueryDone() async {
    if (_exiting) return;
    // Drain any tail text that never hit a sentence boundary (e.g. a
    // model reply ending without final punctuation) so the user hears
    // the whole reply before TTS shuts down.
    if (_ttsRawBuffer.isNotEmpty) {
      final spoken = sanitizeForTts(_ttsRawBuffer);
      _ttsRawBuffer = '';
      if (spoken.isNotEmpty) {
        await widget.voiceService.feedTtsChunk(spoken);
      }
    }
    await widget.voiceService.flushTts();
    final reply = _currentAssistantTurn;
    final sources = _currentAssistantSources;
    if (reply.trim().isNotEmpty) {
      widget.onAssistantTurn(reply, sources);
      // Clear the in-flight buffer now that the message has been committed
      // to chat.messages. Otherwise the captions sheet renders both the
      // committed row AND the still-populated `_currentAssistantTurn`,
      // showing the reply twice during the speaking→listening transition.
      if (mounted) {
        setState(() {
          _currentAssistantTurn = '';
          _currentAssistantSources = null;
        });
      }
    }
    _awaitingPostReplyResume = true;
    if (!widget.voiceService.isSpeaking) {
      // Same grace delay logic as _onSpeakingChanged — audio output
      // buffer needs a moment to drain even when isSpeaking reports false.
      _awaitingPostReplyResume = false;
      Future.delayed(const Duration(milliseconds: 350), () {
        if (_exiting) return;
        if (widget.voiceService.isSpeaking) return;
        _beginUserTurn();
      });
    } else {
      // Defensive: speakingStream sometimes doesn't fire after the last
      // utterance (iOS audio-session interleaving). Arm a watchdog so
      // the user isn't stranded waiting for the mic.
      _resumeWatchdog?.cancel();
      _resumeWatchdog = Timer(const Duration(seconds: 5), () {
        if (_exiting) return;
        if (!_awaitingPostReplyResume) return;
        if (widget.voiceService.isSpeaking) return; // still legitimately busy
        debugPrint('[live-voice] resume watchdog firing — forcing user turn');
        _awaitingPostReplyResume = false;
        _beginUserTurn();
      });
    }
  }

  void _onQueryError(Object error, StackTrace stack) {
    debugPrint('[live-voice] query error: $error\n$stack');
    if (!mounted || _exiting) return;
    setState(() {
      _status = _LiveStatus.error;
      _errorMessage = 'The model errored. Tap retry.';
    });
  }

  void _onSpeakingChanged(bool speaking) {
    if (_exiting) return;
    if (speaking) return;
    if (widget.voiceService.isSpeaking) return;
    final shouldResume =
        _awaitingPostReplyResume || _status == _LiveStatus.speaking;
    if (shouldResume) {
      _resumeWatchdog?.cancel();
      _awaitingPostReplyResume = false;
      // Brief grace delay so the iOS audio output buffer drains before
      // we re-arm the mic. Without this, the last syllable of the AI's
      // reply leaks back into STT as user speech ("...stay still." gets
      // captured as the user saying "stay still"). 350ms is enough for
      // the audio tail without feeling laggy.
      Future.delayed(const Duration(milliseconds: 350), () {
        if (_exiting) return;
        if (widget.voiceService.isSpeaking) return;
        _beginUserTurn();
      });
    }
  }

  /// Camera capture inside live mode. Stops the current listen, snaps a
  /// photo, then runs an image-attached turn through the model with a
  /// "what do you see?" prompt. Skips STT for this turn entirely.
  Future<void> _handleCameraSnap() async {
    if (_exiting) return;
    if (_status == _LiveStatus.thinking || _status == _LiveStatus.speaking) {
      // Don't interrupt an in-flight reply with a camera tap — feels
      // accidental. Require the user to interrupt explicitly first.
      return;
    }
    // Stop listening before opening the picker so the mic doesn't stay
    // hot while the camera UI is up.
    try {
      await widget.voiceService.cancelRecording();
    } catch (_) {}

    final picker = ImagePicker();
    Uint8List? bytes;
    try {
      final image = await picker.pickImage(
        source: ImageSource.camera,
        maxWidth: 768,
        maxHeight: 768,
        imageQuality: 80,
      );
      if (image == null) {
        // User backed out of the camera UI — resume listening cleanly.
        if (!_exiting) await _beginUserTurn();
        return;
      }
      bytes = await image.readAsBytes();
    } catch (e) {
      debugPrint('[live-voice] camera capture failed: $e');
      if (!_exiting) await _beginUserTurn();
      return;
    }
    if (_exiting) return;
    // Mark this as the user's turn (caption sheet picks it up).
    const snapPrompt = 'What do you see in this photo?';
    setState(() {
      _userPartial = '📷 photo';
      _status = _LiveStatus.thinking;
      _currentAssistantTurn = '';
    });
    widget.onUserTurn(snapPrompt);

    // Same auto-extend/trim as the speech turn — image turns also burn
    // context budget (the prompt is short but the engine needs room
    // for the reply tokens).
    await _autoManageContextForLive(snapPrompt);
    if (_exiting) return;

    final priorHistory =
        _isFirstTurn ? widget.priorHistorySnapshot() : const <HistoryTurn>[];
    _isFirstTurn = false;

    try {
      _querySub = widget.service
          .query(
            chatId: widget.chat.id,
            prompt: snapPrompt,
            imageBytes: bytes,
            priorHistory: priorHistory,
            settings: widget.chat.inferenceSettings,
            liveMode: true,
          )
          .listen(_onChunk, onDone: _onQueryDone, onError: _onQueryError);
    } catch (e) {
      debugPrint('[live-voice] camera query failed: $e');
      if (mounted) {
        setState(() {
          _status = _LiveStatus.error;
          _errorMessage = 'Couldn\'t process the photo.';
        });
      }
    }
  }

  /// Run before every live-mode user turn. Estimates how full the
  /// context window will be after this turn replies; if we're close to
  /// the wall, extends memory automatically (or trims if already at the
  /// 32k cap). Awaited so the work folds into the existing "thinking"
  /// pause rather than surfacing as a visible banner — by design, live
  /// mode has no UI for the user to react to mid-conversation.
  ///
  /// Threshold is 85% so we act with headroom: by the time the model
  /// generates its reply (which adds more tokens), we won't have
  /// silently overflowed.
  Future<void> _autoManageContextForLive(String userPrompt) async {
    final est = estimateContextLoad(widget.chat, userPrompt);
    if (est.fraction < 0.85) return;
    final atCap = widget.chat.inferenceSettings.maxTokens >= 32000;
    if (!atCap) {
      // Extend by 2048 and reload. The reload (~5s) happens during the
      // "thinking" state which the user already expects to be a moment
      // of inference latency — no perceptible new pause.
      debugPrint(
          '[live-voice] context ~${est.pct}% — auto-extending memory');
      try {
        await widget.onContextExtend();
      } catch (e) {
        debugPrint('[live-voice] auto-extend failed: $e');
      }
    } else {
      // Already at the engine's maximum. Drop the oldest ~30% of turns
      // so the reply has room. The trim is silent — no engine reload,
      // just a chat-session reset (resetChatSession), so the next
      // query replays the trimmed history into a fresh KV cache.
      debugPrint(
          '[live-voice] context ~${est.pct}% at 32k cap — auto-trimming');
      try {
        await widget.onContextTrim();
      } catch (e) {
        debugPrint('[live-voice] auto-trim failed: $e');
      }
    }
  }

  /// Tap-to-retry from the error state. (Interrupt-mid-reply was removed —
  /// the iOS audio session juggling needed to stop TTS, halt the model,
  /// and re-arm the mic was a chronic crash source. The user must now let
  /// the reply finish; if it's truly stuck they can End and re-enter.)
  Future<void> _retryFromError() async {
    if (_status != _LiveStatus.error) return;
    setState(() {
      _errorMessage = null;
      _status = _LiveStatus.initializing;
    });
    await _start();
  }

  Future<void> _exit() async {
    if (_exiting) return; // idempotent — back tap can fire twice quickly
    _exiting = true;
    _resumeWatchdog?.cancel();
    _sttWatchdog?.cancel();

    // 1. Stop our handlers from firing first so nothing reads stale state.
    await _partialSub?.cancel();
    await _finalSub?.cancel();
    await _speakingSub?.cancel();
    await _errorSub?.cancel();
    await _querySub?.cancel();

    // 2. Stop audio I/O. Both are well-behaved on iOS.
    try {
      await widget.voiceService.cancelRecording();
    } catch (_) {}
    try {
      await widget.voiceService.stopSpeaking();
    } catch (_) {}

    // 3. Hand control back to the host BEFORE we attempt to stop the
    // model. This is the change that fixes the "back-tap during TTS
    // playback crashes the app" bug: by the time TTS is draining the
    // bottom of a reply, the model has almost always finished
    // generating — but `_persistedChat.stopGeneration()` on a completed
    // chat segfaults the iOS FFI, and native segfaults can't be caught
    // in Dart. The service-side flag below also short-circuits stop()
    // when nothing is actively generating.
    widget.onExit();

    // 4. Best-effort model stop, fire-and-forget. service.stop() itself
    // is now defensive — it no-ops when `isGenerating` is false. Any
    // remaining FFI risk happens AFTER the user has navigated away.
    if (widget.service.isGenerating) {
      // Schedule on next microtask so the screen-removal can settle
      // before we touch native code.
      Future.microtask(() async {
        try {
          await widget.service.stop();
        } catch (e) {
          debugPrint('[live-voice] post-exit service.stop failed: $e');
        }
      });
    }
  }

  // ──────────────────────────────────────────────────────────────
  // Visual helpers — derive orb appearance from current state
  // ──────────────────────────────────────────────────────────────

  Color _orbColor(AshColors c) {
    switch (_status) {
      case _LiveStatus.listening:
        return c.accent;
      case _LiveStatus.thinking:
        return c.accent;
      case _LiveStatus.speaking:
        return c.accent;
      case _LiveStatus.error:
        return Colors.redAccent;
      case _LiveStatus.initializing:
        return c.textMuted;
    }
  }

  String _statusLabel() {
    switch (_status) {
      case _LiveStatus.initializing:
        return 'Connecting…';
      case _LiveStatus.listening:
        return 'Listening';
      case _LiveStatus.thinking:
        return 'Thinking';
      case _LiveStatus.speaking:
        return 'Speaking';
      case _LiveStatus.error:
        return 'Error';
    }
  }

  // ──────────────────────────────────────────────────────────────
  // Build
  // ──────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final c = ash(context);
    return PopScope(
      canPop: false,
      onPopInvokedWithResult: (didPop, _) {
        if (!didPop) _exit();
      },
      child: Scaffold(
        backgroundColor: c.bg,
        body: SafeArea(
          child: Stack(
            children: [
              // Soft radial backdrop tinted by the orb color so the whole
              // screen subtly breathes with the state.
              Positioned.fill(child: _backdrop(c)),
              // Primary surface (header / orb / controls).
              Column(
                children: [
                  _header(c),
                  Expanded(child: _orb(c)),
                  _statusLine(c),
                  _controlBar(c),
                  const SizedBox(height: 16),
                ],
              ),
              // Captions overlay slides up from the bottom when toggled.
              if (_showCaptions) _captionsSheet(c),
            ],
          ),
        ),
      ),
    );
  }

  Widget _backdrop(AshColors c) {
    return AnimatedBuilder(
      animation: _slowPulse,
      builder: (context, _) {
        final t = _slowPulse.value;
        return DecoratedBox(
          decoration: BoxDecoration(
            gradient: RadialGradient(
              center: const Alignment(0, -0.2),
              radius: 1.2,
              colors: [
                _orbColor(c).withValues(alpha: 0.08 + 0.04 * t),
                c.bg,
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _header(AshColors c) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(12, 8, 12, 0),
      child: Row(
        children: [
          _RoundButton(
            icon: Icons.close,
            onTap: _exit,
            background: c.surfaceElev,
            iconColor: c.text,
            size: 40,
          ),
          const Spacer(),
          // Captions toggle (top-right, like ChatGPT's CC button).
          _RoundButton(
            icon:
                _showCaptions ? Icons.closed_caption : Icons.closed_caption_off,
            onTap: () => setState(() => _showCaptions = !_showCaptions),
            background: _showCaptions ? c.accent : c.surfaceElev,
            iconColor: _showCaptions ? Colors.white : c.text,
            size: 40,
          ),
        ],
      ),
    );
  }

  Widget _orb(AshColors c) {
    final color = _orbColor(c);
    return Center(
      child: GestureDetector(
        onTap: _status == _LiveStatus.error ? _retryFromError : null,
        behavior: HitTestBehavior.opaque,
        child: AnimatedBuilder(
          animation: Listenable.merge([_slowPulse, _fastPulse]),
          builder: (context, _) {
            // Outer halo amplitude depends on state — bigger / faster while
            // the model is actively speaking, smaller when listening, mostly
            // still when initializing.
            final pulseT = switch (_status) {
              _LiveStatus.speaking => _fastPulse.value,
              _LiveStatus.listening => _fastPulse.value * 0.6,
              _LiveStatus.thinking => _slowPulse.value * 0.8,
              _LiveStatus.error => 0.0,
              _LiveStatus.initializing => _slowPulse.value * 0.4,
            };
            return SizedBox(
              width: 320,
              height: 320,
              child: Stack(
                alignment: Alignment.center,
                children: [
                  // Outer expanding halo — ripple effect during speaking.
                  if (_status == _LiveStatus.speaking ||
                      _status == _LiveStatus.listening)
                    _Halo(progress: _fastPulse.value, color: color),
                  // Core glow disc.
                  Container(
                    width: 200 + pulseT * 40,
                    height: 200 + pulseT * 40,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      gradient: RadialGradient(
                        colors: [
                          color.withValues(alpha: 0.85),
                          color.withValues(alpha: 0.0),
                        ],
                      ),
                    ),
                  ),
                  // Thinking indicator — a thin Material progress ring
                  // sitting just outside the inner disc. Replaces the
                  // earlier SweepGradient+dstOut mask, which rendered as
                  // a static dark void (the mask cleared the gradient's
                  // 70%-transparent region down to bg, and the FFI
                  // prefill briefly froze the spin animation on top of
                  // that). CircularProgressIndicator is a fixed-stroke
                  // arc that always reads as a spinner, even when the
                  // UI thread is briefly busy.
                  if (_status == _LiveStatus.thinking)
                    SizedBox(
                      width: 230,
                      height: 230,
                      child: CircularProgressIndicator(
                        strokeWidth: 3,
                        valueColor: AlwaysStoppedAnimation<Color>(
                          color.withValues(alpha: 0.85),
                        ),
                        backgroundColor: Colors.transparent,
                      ),
                    ),
                  // Inner solid disc with subtle pulse.
                  Container(
                    width: 140 + pulseT * 12,
                    height: 140 + pulseT * 12,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: color,
                      boxShadow: [
                        BoxShadow(
                          color: color.withValues(alpha: 0.5),
                          blurRadius: 40,
                          spreadRadius: pulseT * 8,
                        ),
                      ],
                    ),
                    child: const Center(
                      child: Icon(
                        Icons.graphic_eq,
                        color: Colors.white,
                        size: 36,
                      ),
                    ),
                  ),
                ],
              ),
            );
          },
        ),
      ),
    );
  }

  Widget _statusLine(AshColors c) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 24),
      child: AnimatedSwitcher(
        duration: const Duration(milliseconds: 200),
        child: Column(
          key: ValueKey(_status),
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              _statusLabel(),
              style: TextStyle(
                color: c.text,
                fontSize: 15,
                fontWeight: FontWeight.w600,
              ),
            ),
            if (_status == _LiveStatus.listening && _userPartial.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(top: 6),
                child: Text(
                  _userPartial,
                  textAlign: TextAlign.center,
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(color: c.textMuted, fontSize: 13),
                ),
              ),
            if (_status == _LiveStatus.error && _errorMessage != null)
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: Text(
                  _errorMessage!,
                  textAlign: TextAlign.center,
                  style: TextStyle(color: c.textMuted, fontSize: 13),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _controlBar(AshColors c) {
    // Two buttons: Camera • End. Stop/interrupt and mic-mute were removed
    // because the iOS audio session juggling needed to halt TTS, stop the
    // model, and re-arm the mic was a chronic crash source
    // (AVAudioSession state corruption across cycles → SIGSEGV inside
    // SpeechToTextPlugin.listenForSpeech). The conversation now has only
    // one path: listen → reply → listen, with End to leave.
    final isReplying = _status == _LiveStatus.thinking ||
        _status == _LiveStatus.speaking;
    final canSnap = !isReplying && _status != _LiveStatus.initializing;
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: [
          _RoundButton(
            icon: Icons.camera_alt_outlined,
            onTap: canSnap ? _handleCameraSnap : null,
            background: c.surfaceElev,
            iconColor:
                canSnap ? c.text : c.textDim.withValues(alpha: 0.5),
            size: 52,
          ),
          _RoundButton(
            icon: Icons.call_end,
            onTap: _exit,
            background: Colors.redAccent,
            iconColor: Colors.white,
            size: 52,
          ),
        ],
      ),
    );
  }

  Widget _captionsSheet(AshColors c) {
    final messages = widget.chat.messages;
    return Positioned(
      left: 0,
      right: 0,
      bottom: 0,
      child: AnimatedSlide(
        offset: _showCaptions ? Offset.zero : const Offset(0, 1),
        duration: const Duration(milliseconds: 220),
        curve: Curves.easeOutCubic,
        child: Container(
          height: MediaQuery.of(context).size.height * 0.55,
          decoration: BoxDecoration(
            color: c.surfaceElev,
            borderRadius:
                const BorderRadius.vertical(top: Radius.circular(24)),
            border: Border(
              top: BorderSide(color: c.border, width: 0.5),
            ),
          ),
          child: Column(
            children: [
              // Drag handle
              Container(
                width: 36,
                height: 4,
                margin: const EdgeInsets.only(top: 10, bottom: 10),
                decoration: BoxDecoration(
                  color: c.textDim,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
                child: Row(
                  children: [
                    Text(
                      'Transcript',
                      style: TextStyle(
                        color: c.text,
                        fontSize: 14,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    const Spacer(),
                    GestureDetector(
                      onTap: () => setState(() => _showCaptions = false),
                      child: Icon(Icons.close, color: c.textMuted, size: 20),
                    ),
                  ],
                ),
              ),
              Divider(height: 1, color: c.border),
              Expanded(
                child: ListView.separated(
                  reverse: true,
                  padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
                  itemCount: _transcriptCount(messages),
                  separatorBuilder: (_, __) => const SizedBox(height: 12),
                  itemBuilder: (context, i) {
                    // Reverse so newest is at the bottom (closest to user)
                    // and grows upward. Also append the in-progress
                    // assistant turn (not yet committed to chat.messages) at
                    // index 0 so the user sees streaming text live.
                    final showLive = _currentAssistantTurn.isNotEmpty &&
                        _status == _LiveStatus.speaking;
                    if (i == 0 && showLive) {
                      // The in-flight turn streams chunk-by-chunk. Rendering
                      // it as Markdown while the markdown is incomplete
                      // (mid-token, unclosed `**`, fragmentary `<think>`)
                      // can throw inside flutter_markdown_plus's render
                      // pipeline and propagate as a native crash. Plain
                      // Text is safe and the visual difference for a
                      // streaming reply is negligible.
                      return _captionRow(
                        c,
                        text: _currentAssistantTurn,
                        isUser: false,
                        renderAsPlainText: true,
                      );
                    }
                    final adjusted = showLive ? i - 1 : i;
                    final idx = messages.length - 1 - adjusted;
                    if (idx < 0) return const SizedBox.shrink();
                    final m = messages[idx];
                    return _captionRow(
                      c,
                      text: m.text,
                      isUser: m.role == 'user',
                    );
                  },
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  int _transcriptCount(List<dynamic> messages) {
    final showLive = _currentAssistantTurn.isNotEmpty &&
        _status == _LiveStatus.speaking;
    return messages.length + (showLive ? 1 : 0);
  }

  Widget _captionRow(AshColors c,
      {required String text,
      required bool isUser,
      bool renderAsPlainText = false}) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          width: 28,
          height: 28,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: isUser ? c.accent : c.surfaceElev,
            border: Border.all(color: c.border, width: 0.5),
          ),
          child: Center(
            child: Icon(
              isUser ? Icons.person : Icons.auto_awesome,
              color: isUser ? Colors.white : c.accent,
              size: 14,
            ),
          ),
        ),
        const SizedBox(width: 10),
        Expanded(
          child: (isUser || renderAsPlainText)
              ? Text(
                  text,
                  style: TextStyle(color: c.text, fontSize: 14, height: 1.4),
                )
              : MarkdownBody(
                  data: text,
                  selectable: true,
                  softLineBreak: true,
                  styleSheet: MarkdownStyleSheet(
                    p: TextStyle(
                        color: c.text, fontSize: 14, height: 1.4),
                    strong:
                        TextStyle(color: c.text, fontWeight: FontWeight.w700),
                    em: TextStyle(
                        color: c.text, fontStyle: FontStyle.italic),
                    code: TextStyle(
                      color: c.accent,
                      fontSize: 13,
                      fontFamily: 'Menlo',
                      backgroundColor:
                          c.surfaceElev.withValues(alpha: 0.6),
                    ),
                  ),
                ),
        ),
      ],
    );
  }
}

// ─── Halo (single ripple) ────────────────────────────────────────────────────

class _Halo extends StatelessWidget {
  const _Halo({required this.progress, required this.color});
  final double progress;
  final Color color;

  @override
  Widget build(BuildContext context) {
    final size = 200 + progress * 140;
    return IgnorePointer(
      child: Container(
        width: size,
        height: size,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          border: Border.all(
            color: color.withValues(alpha: 0.4 * (1 - progress)),
            width: 2,
          ),
        ),
      ),
    );
  }
}

// ─── Round button used for Close / Captions / Mic / End ──────────────────────

class _RoundButton extends StatelessWidget {
  const _RoundButton({
    required this.icon,
    required this.onTap,
    required this.background,
    required this.iconColor,
    required this.size,
  });

  final IconData icon;
  /// Null = button is visually shown but inert. Used by the interrupt
  /// button: it lives in the control bar all the time, but only fires
  /// when the assistant is thinking or speaking.
  final VoidCallback? onTap;
  final Color background;
  final Color iconColor;
  final double size;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: size,
        height: size,
        decoration: BoxDecoration(
          color: background,
          shape: BoxShape.circle,
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.25),
              blurRadius: 6,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: Icon(icon, color: iconColor, size: size * 0.45),
      ),
    );
  }
}
