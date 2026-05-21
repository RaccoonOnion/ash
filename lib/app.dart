import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'models/app_state.dart';
import 'models/chat.dart';
import 'models/pack.dart';
import 'services/context_estimator.dart';
import 'services/model_download_state.dart';
import 'screens/onboarding_screen.dart';
import 'screens/model_pick_screen.dart';
import 'screens/model_download_screen.dart';
import 'screens/home_screen.dart';
import 'screens/chat_screen.dart';
import 'screens/knowledge_screen.dart';
import 'screens/pack_reader_screen.dart';
import 'screens/live_voice_screen.dart';
import 'screens/profile_screen.dart';
import 'screens/models_screen.dart';
import 'screens/settings_screen.dart';
import 'services/inference_service.dart';
// MessageSource is exported via inference_service.dart for the citation-chip
// payload.
import 'services/llm_model.dart';
import 'services/voice_service.dart';
import 'theme/ash_theme.dart';
import 'widgets/tab_bar.dart' as ash;
import 'widgets/buttons.dart';
import 'widgets/progress_ring.dart';
import 'widgets/glass.dart';

// Fresh installs start with an empty chat list; the home screen's empty
// state surfaces example prompts that turn into real chats on tap.
final List<Chat> _seedChats = const <Chat>[];

// ─── AshApp ──────────────────────────────────────────────────────────────────

class AshApp extends StatefulWidget {
  const AshApp({
    super.key,
    required this.service,
    required this.voiceService,
  });

  final InferenceService service;
  final VoiceService voiceService;

  @override
  State<AshApp> createState() => _AshAppState();
}

class _AshAppState extends State<AshApp> {
  // ── Stage & navigation ───────────────────────────────────────
  final _navigatorKey = GlobalKey<NavigatorState>();
  // Default to onboarding, but initState() consults SharedPreferences
  // and may fast-skip to modelPick or main if the user has already
  // completed the carousel on a previous launch (and, in the latter
  // case, also has a model installed). The flag we read is
  // `onboarding_carousel_done` — written in OnboardingScreen.onDone.
  AppStage _stage = AppStage.onboarding;
  MainTab _tab = MainTab.home;
  String? _chatId;
  static const _kPrefOnboardingDone = 'onboarding_carousel_done';

  // ── Data ─────────────────────────────────────────────────────
  List<Pack> _packs = [];
  List<Chat> _chats = _seedChats;

  /// One-shot wipe for the round-5 proposition pipeline. The new pack JSONs
  /// hold LLM-extracted Q&A propositions (question stored as the first
  /// `####` heading in each chunk's text, answer following). Embeddings
  /// represent the QUESTION — which aligns much better with user-typed
  /// queries than the previous "raw markdown paragraph" embeddings.
  /// Bumped to `embeddingModelVersion: "all-MiniLM-L6-v2-r5"`.
  ///
  /// On-device ObjectBox still has the round-4 chunks; we wipe + reimport
  /// once when the marker is missing. Marker survives across launches.
  ///
  /// Round 5b: build-time `polish_answer` cleanup landed (lone-numbered
  /// demote, blockquote unwrap, hazadapt:// link strip, etc.). Embeddings
  /// didn't change — the polish only touches the answer text — but
  /// devices that already have round-5 chunks still hold the old answer
  /// strings in ObjectBox. A second marker fires the wipe once more.
  Future<void> _runRound5PropositionsMigrationIfNeeded(
    Set<String> installedIds,
  ) async {
    try {
      final docsDir = await getApplicationDocumentsDirectory();
      final flagFile =
          File('${docsDir.path}/rag_core/.round5b_polish_done');
      if (await flagFile.exists()) return;
      if (installedIds.isNotEmpty) {
        // Snapshot what the user had before wiping — round-5b changed the
        // underlying chunk text format, but the user's intent (which
        // packs they care about) is preserved. After the wipe we
        // re-import EVERY previously-installed pack with the new data,
        // not just the essentials. Without this snapshot/restore the
        // user silently lost any non-essential packs they'd added (the
        // exact regression they reported: "i got shot" stopped grounding
        // because gunshot-wound got wiped and never re-imported).
        final previouslyInstalled = installedIds.toSet();
        debugPrint('[ash] round-5b polish migration: '
            'wiping ${previouslyInstalled.length} pack(s) to swap in '
            'cleaned-up text, then re-importing each one');
        for (final id in previouslyInstalled) {
          await widget.service.uninstallPack(id);
        }
        installedIds.clear();
        // Re-import each previously-installed pack. importPack() reads
        // the asset bundle (which now has round-5 propositions) and
        // populates ObjectBox.
        for (final id in previouslyInstalled) {
          await widget.service.importPack(id);
          installedIds.add(id);
        }
      }
      await flagFile.parent.create(recursive: true);
      await flagFile.create();
    } catch (e) {
      debugPrint('[ash] round-5b migration failed (non-fatal): $e');
    }
  }

  /// One-shot wipe for the round-4 re-chunk + re-embed. The new pack JSONs
  /// have 3x more chunks at finer granularity and were re-embedded with a
  /// bumped `embeddingModelVersion` ("all-MiniLM-L6-v2-r4"). The on-device
  /// ObjectBox still has the old coarse chunks, and `importPack` is a
  /// putMany that would append duplicates rather than swap. So: drop every
  /// installed pack once. Essentials re-import on the next loop in
  /// [_loadPacks], custom user-installed packs need a re-install from the
  /// Store (acceptable for the chunking quality gain).
  ///
  /// Idempotent — marker survives across launches until the app is deleted.
  Future<void> _runRound4RechunkMigrationIfNeeded(
    Set<String> installedIds,
  ) async {
    try {
      final docsDir = await getApplicationDocumentsDirectory();
      final flagFile =
          File('${docsDir.path}/rag_core/.round4_rechunked_done');
      if (await flagFile.exists()) return;
      if (installedIds.isNotEmpty) {
        debugPrint('[ash] round-4 rechunk migration: '
            'wiping ${installedIds.length} pack(s) so the new embeddings '
            'replace the old chunks');
        for (final id in installedIds.toList()) {
          await widget.service.uninstallPack(id);
        }
        installedIds.clear();
      }
      await flagFile.parent.create(recursive: true);
      await flagFile.create();
    } catch (e) {
      debugPrint('[ash] round-4 migration failed (non-fatal): $e');
    }
  }

  /// One-shot scrub for devices that came from Round 1 (when launch
  /// blanket-installed every pack). When the marker file is absent and we
  /// see more than just essentials in the vector store, uninstall the
  /// non-essentials in place. Mutates [installedIds] to reflect the new
  /// state so the rest of [_loadPacks] sees the right snapshot. Idempotent
  /// — after the marker drops, this is a no-op forever.
  Future<void> _runLegacyInstallScrubIfNeeded(
    Set<String> installedIds,
    List<Pack> registry,
  ) async {
    try {
      final docsDir = await getApplicationDocumentsDirectory();
      final flagFile =
          File('${docsDir.path}/rag_core/.essentials_migration_done');
      if (await flagFile.exists()) return;

      final essentialIds = registry
          .where((p) => p.tier == PackTier.essential)
          .map((p) => p.id)
          .toSet();
      final extras = installedIds
          .where((id) => !essentialIds.contains(id))
          .toList(growable: false);
      if (extras.isNotEmpty) {
        debugPrint('[ash] essentials migration: scrubbing '
            '${extras.length} non-essential packs from legacy state');
        for (final id in extras) {
          await widget.service.uninstallPack(id);
        }
        installedIds.removeAll(extras);
      }
      // Marker stays in the ObjectBox dir — survives reinstalls of the same
      // app build, but a TestFlight wipe / "Delete App" clears it (which is
      // the right behavior: a clean app should re-evaluate).
      await flagFile.parent.create(recursive: true);
      await flagFile.create();
    } catch (e) {
      debugPrint('[ash] essentials migration failed (non-fatal): $e');
    }
  }

  Future<void> _loadPacks() async {
    try {
      final jsonStr = await rootBundle.loadString(
        'assets/rag/packs/packs_registry.json',
      );
      final loaded = packsFromRegistry(jsonStr);
      if (!mounted) return;

      // Honor prior install state across launches — ObjectBox is durable, so
      // a pack the user explicitly installed in a previous session shows up
      // already-installed. Round 1 just blanket-marked everything installed;
      // Round 2 reads truth from the vector store.
      final installedIds =
          (await widget.service.installedPackIds()).toSet();

      // Round 5 first (most recent migration): wipe + re-import so the
      // question-embedded proposition chunks take over. Subsumes both
      // round-2 scrub and round-4 wipe when those are still pending.
      await _runRound5PropositionsMigrationIfNeeded(installedIds);

      // Round 4: if the new re-chunked + re-embedded packs haven't
      // taken over yet, wipe everything so essentials re-import below
      // pulls the new chunks. (Subsumes the round-2 scrub when both
      // migrations are pending — the wipe is more aggressive.)
      await _runRound4RechunkMigrationIfNeeded(installedIds);

      // One-shot migration: devices that ran Round 1 have ALL 55 packs in
      // ObjectBox. The new contract is "only essentials auto-install"; we
      // need to scrub the rest exactly once so the Library/Store split
      // reflects reality. Flag-file lives in the app docs dir.
      await _runLegacyInstallScrubIfNeeded(installedIds, loaded);

      // First-run convenience: auto-import the Essential tier so a fresh
      // device has a useful starter library without forcing the user to
      // hunt through the Store. Everything else stays unset until the user
      // taps Install.
      for (final p in loaded) {
        if (p.tier == PackTier.essential && !installedIds.contains(p.id)) {
          await widget.service.importPack(p.id);
          installedIds.add(p.id);
        }
      }

      final packsWithState = loaded
          .map((p) => p.copyWith(installed: installedIds.contains(p.id)))
          .toList();

      setState(() => _packs = packsWithState);
      // Default retrieval lens for fresh chats: all installed (null lens).
      // Per-chat scoping happens when the user opens a chat with a non-null
      // lensPackIds (e.g. "Ask Ash about this section" from the Library).
      widget.service.setLens(null);
    } catch (e) {
      debugPrint('[ash] Failed to load packs: $e');
    }
  }

  // ── Sheets ───────────────────────────────────────────────────
  String? _sheet; // null, 'voice', 'camera', 'settings'

  // ── Model download ───────────────────────────────────────────
  // Per-model download state. Each variant owns its lifecycle —
  // idle/downloading/paused/installed/failed — so concurrent downloads
  // don't collide and pause/resume can be wired to a per-variant
  // CancelToken. The bottom banner and download screen read whichever
  // variant the user most recently picked (`_selectedModel`); the
  // Models tab renders every entry.
  final Map<LlmModel, ModelDownloadInfo> _downloads =
      <LlmModel, ModelDownloadInfo>{};

  /// Live ETA estimator per active download. Lazily allocated on
  /// `_startDownloadFor`, cleared on completion / cancel / fail. Kept off
  /// `ModelDownloadInfo` itself because the estimator is mutable state
  /// and the info struct is supposed to be a copy-friendly snapshot.
  final Map<LlmModel, EtaEstimator> _etaEstimators =
      <LlmModel, EtaEstimator>{};
  int _downloadEta = 0;
  ModelInfo? _selectedModel;
  bool _modelReady = false;

  ModelDownloadInfo _infoFor(LlmModel m) =>
      _downloads[m] ?? const ModelDownloadInfo();

  /// The first variant that's currently downloading (state ==
  /// downloading). Null when none is in flight. Drives the bottom banner.
  LlmModel? get _downloadingVariant {
    for (final entry in _downloads.entries) {
      if (entry.value.state == DownloadState.downloading) return entry.key;
    }
    return null;
  }

  /// Whether any variant is in `downloading` state right now. Used by the
  /// floating banner so it stays visible while ANY install is in flight,
  /// regardless of which one is foregrounded.
  bool get _isAnyDownloading => _downloadingVariant != null;

  /// Progress shown by the foregrounded download UI (ModelDownloadScreen,
  /// download banner) — the variant the user most recently picked.
  double get _selectedDownloadProgress {
    final m = _selectedModel == null
        ? null
        : LlmModel.fromId(_selectedModel!.id);
    return m == null ? 0 : _infoFor(m).progress;
  }

  /// Mutator helper — copy-with on the per-model entry. Always calls
  /// setState; callers don't need to wrap.
  void _updateDownload(
    LlmModel m, {
    DownloadState? state,
    double? progress,
    int? etaMin,
    CancelToken? cancelToken,
    String? errorMessage,
    bool clearCancelToken = false,
    bool clearError = false,
  }) {
    final current = _infoFor(m);
    final next = current.copyWith(
      state: state,
      progress: progress,
      etaMin: etaMin,
      cancelToken: cancelToken,
      errorMessage: errorMessage,
      clearCancelToken: clearCancelToken,
      clearError: clearError,
    );
    setState(() => _downloads[m] = next);
  }

  // ── Inference ────────────────────────────────────────────────
  bool _isGenerating = false;
  bool _isWarmingUp = false;
  // Variants present on disk. Updated after install/delete or refreshed
  // from the service on demand. Used to drive the model-picker UI in
  // Settings — uninstalled variants get a "Download" badge instead of "Switch".
  Set<LlmModel> _installedLlmModels = {};
  // Non-null while a query is waiting on a reconfiguration step (engine
  // reload from maxTokens change, vision-encoder load on first image, or
  // chat-session rebuild from sampling/RAG change). Drives _ReconfigureBanner
  // so the user knows why the first token is slow. Cleared on first chunk.
  _ReconfigureStatus? _reconfigureStatus;

  // ── Pending image attachment ────────────────────────────────
  Uint8List? _pendingImageBytes;

  static const _huggingFaceToken = String.fromEnvironment('HUGGINGFACE_TOKEN');

  @override
  void initState() {
    super.initState();
    _loadPacks();
    _kickOffWarmUpIfReady();
    _refreshInstalledLlmModels();
    _maybeSkipOnboarding();
  }

  /// On launch, consult [_kPrefOnboardingDone]. If the user has seen the
  /// carousel before, skip straight past it: to Main if a model is
  /// already installed, otherwise to ModelPick. The model-pick and
  /// download stages are state-driven and will still run as needed —
  /// the only thing this flag suppresses is the 3-slide explainer.
  Future<void> _maybeSkipOnboarding() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final done = prefs.getBool(_kPrefOnboardingDone) ?? false;
      if (!done || !mounted) return;
      // The user has seen the carousel. Decide where to land based on
      // whether a model is on disk. _refreshInstalledLlmModels() is
      // async and may not have finished yet — poke the service
      // directly. A momentary flash through ModelPick is acceptable if
      // we get the race wrong; the destructive case (sending a
      // first-launch user past the carousel) can't happen because the
      // pref defaults to false.
      final hasModel = await _anyModelOnDisk();
      if (!mounted) return;
      setState(() {
        _stage = hasModel ? AppStage.main : AppStage.modelPick;
      });
    } catch (e) {
      debugPrint('[ash] onboarding skip check failed: $e');
    }
  }

  Future<bool> _anyModelOnDisk() async {
    for (final m in LlmModel.values) {
      try {
        final s = await widget.service.getModelStatus(model: m);
        if (s.isInstalled) return true;
      } catch (_) {}
    }
    return false;
  }

  Future<void> _markOnboardingDone() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setBool(_kPrefOnboardingDone, true);
    } catch (e) {
      debugPrint('[ash] mark onboarding done failed: $e');
    }
  }

  /// Warm the on-device model as early as possible so first-query latency
  /// (especially for image queries, where the vision encoder Metal JIT can take
  /// 30-60 s) is paid up-front while the user is still in onboarding/login.
  /// Probe every known variant to see which `.litertlm` files are on disk.
  /// Cheap (a few SDK calls); run on launch + after install/delete.
  Future<void> _refreshInstalledLlmModels() async {
    final installed = <LlmModel>{};
    for (final m in LlmModel.values) {
      try {
        final s = await widget.service.getModelStatus(model: m);
        if (s.isInstalled) installed.add(m);
      } catch (_) {
        // Swallow — a single variant failing to probe shouldn't break the
        // others. Worst case: it'll show as "Download" until next launch.
      }
    }
    if (!mounted) return;
    setState(() => _installedLlmModels = installed);
  }

  /// Flip the MTP / speculative-decoding flag. Same shape as accelerator
  /// or model swap — engine closes + re-warms; the reconfigure banner
  /// shows during the rebuild.
  Future<void> _toggleSpeculativeDecoding(bool enabled) async {
    if (enabled == widget.service.speculativeDecoding) return;
    setState(() => _reconfigureStatus = _ReconfigureStatus.engineReload);
    try {
      await widget.service.setSpeculativeDecoding(enabled);
      await widget.service.warmUp();
    } catch (e, st) {
      debugPrint('[ash] setSpeculativeDecoding failed: $e\n$st');
    } finally {
      if (mounted) setState(() => _reconfigureStatus = null);
    }
  }

  /// Switch the active variant. Shows the engine-reload banner during the
  /// close+re-warm so the user knows why the next message has a leading wait.
  Future<void> _switchActiveLlmModel(LlmModel target) async {
    if (target == widget.service.activeLlmModel) return;
    setState(() => _reconfigureStatus = _ReconfigureStatus.engineReload);
    try {
      await widget.service.setActiveLlmModel(target);
      // Engine was closed by setActiveLlmModel — re-warm so the next query
      // streams without cold-load latency. _kickOffWarmUpIfReady will set
      // _isWarmingUp/clear it on completion.
      await widget.service.warmUp();
    } catch (e, st) {
      debugPrint('[ash] active model switch failed: $e\n$st');
    } finally {
      if (mounted) setState(() => _reconfigureStatus = null);
    }
  }

  /// Look up the [ModelInfo] (seedModels card) that matches a given
  /// [LlmModel]. Used when the user picks a variant for download from the
  /// Settings sheet — the download stage expects [_selectedModel] to be set.
  ModelInfo? _findModelInfoFor(LlmModel m) {
    for (final info in seedModels) {
      if (LlmModel.fromId(info.id) == m) return info;
    }
    return null;
  }

  /// Remove a variant's `.litertlm` from disk.
  ///
  /// Deletion semantics:
  /// - If [m] is the currently-active variant, we switch to another
  ///   installed variant FIRST (so the engine isn't pointing at a file
  ///   that's about to disappear), then delete.
  /// - If [m] is the active variant AND no other variants are installed,
  ///   we delete it and route the app back to [AppStage.modelPick] so the
  ///   user can pick + download a replacement. The chat surface can't
  ///   meaningfully exist without a model.
  Future<void> _deleteLlmModel(LlmModel m) async {
    final wasActive = m == widget.service.activeLlmModel;

    // Hand-off BEFORE delete so the engine doesn't briefly hold a handle
    // to a file we're about to remove.
    if (wasActive) {
      final fallback = _installedLlmModels
          .firstWhere((other) => other != m, orElse: () => m);
      if (fallback != m) {
        try {
          await _switchActiveLlmModel(fallback);
        } catch (e, st) {
          debugPrint('[ash] auto-switch before delete failed: $e\n$st');
        }
      }
    }

    try {
      await widget.service.deleteModel(model: m);
    } catch (e, st) {
      debugPrint('[ash] deleteModel($m) failed: $e\n$st');
      // Even on failure, refresh so UI doesn't lie about disk state.
    }
    await _refreshInstalledLlmModels();

    // If we deleted the last installed model, send the user back to
    // ModelPick so they have an obvious path to recovery.
    if (!mounted) return;
    if (wasActive && _installedLlmModels.isEmpty) {
      setState(() {
        _stage = AppStage.modelPick;
        _chatId = null;
        _sheet = null;
        _selectedModel = null;
      });
    }
  }

  /// Wipe every chat from memory. We don't currently persist chat history,
  /// so this is purely an in-memory reset — closing/relaunching also wipes
  /// chats. Resets active chat ID and the next-chat counter.
  void _clearAllConversations() {
    setState(() {
      _chats = [];
      _chatId = null;
      _sheet = null;
    });
  }

  Future<void> _kickOffWarmUpIfReady() async {
    if (widget.service.isWarm || _isWarmingUp) return;
    try {
      final status = await widget.service.getModelStatus();
      if (!status.isInstalled) return;
      if (widget.service.isWarm || _isWarmingUp) return;
      if (!mounted) return;
      setState(() => _isWarmingUp = true);
      await widget.service.warmUp();
      if (!mounted) return;
      setState(() => _isWarmingUp = false);
    } catch (e) {
      debugPrint('[ash] warm-up failed: $e');
      if (!mounted) return;
      setState(() => _isWarmingUp = false);
    }
  }

  // ── Helpers ──────────────────────────────────────────────────

  List<Pack> get _installedPacks =>
      _packs.where((p) => p.installed).toList(growable: false);

  Chat? get _activeChat =>
      _chatId != null ? _chats.where((c) => c.id == _chatId).firstOrNull : null;

  /// Single source of truth for "can the user start an inference now?".
  /// False while an install is in flight (so we don't kick off engine_create
  /// against a half-written file) AND when the active variant simply isn't
  /// on disk yet. Every entry point that can trigger inference (chat send,
  /// image send, live voice mode) consults this; the chat composer also
  /// presents a disabled-but-friendly state when it's false so the user
  /// can compose ahead.
  bool get _modelOnDisk {
    final active = widget.service.activeLlmModel;
    final info = _infoFor(active);
    if (info.state == DownloadState.downloading) return false;
    return _installedLlmModels.contains(active);
  }

  // Transient banner shown when the user pokes a model-gated control while
  // the model is still downloading. Auto-clears.
  String? _waitingForModelToast;
  Timer? _waitingForModelToastTimer;

  void _showWaitingForModel([String? message]) {
    _waitingForModelToastTimer?.cancel();
    setState(() => _waitingForModelToast =
        message ?? 'Ash is still downloading. Hang tight — you can chat as soon as it lands.');
    _waitingForModelToastTimer = Timer(const Duration(seconds: 3), () {
      if (mounted) setState(() => _waitingForModelToast = null);
    });
  }

  void _setStage(AppStage s) => setState(() => _stage = s);

  int _nextChatId() {
    final maxId = _chats
        .map((c) => int.tryParse(c.id.replaceFirst('c', '')) ?? 0)
        .fold(0, math.max);
    return maxId + 1;
  }

  // ── Real model download ──────────────────────────────────────

  /// Onboarding/legacy entry point — picks the model from `_selectedModel`
  /// and delegates to [_startDownloadFor]. Settings + Models tab call
  /// `_startDownloadFor` directly so they don't have to round-trip through
  /// `_selectedModel`.
  void _startDownload() {
    final selectedLlm = _selectedModel == null
        ? null
        : LlmModel.fromId(_selectedModel!.id);
    if (selectedLlm == null) return;
    _startDownloadFor(selectedLlm);
  }

  /// Begin (or resume) downloading [m]. Idempotent — calling again while
  /// the same variant is in `downloading` is a no-op; while it's `paused`
  /// triggers a resume; on `installed` switches the active model.
  void _startDownloadFor(LlmModel m) {
    final info = _infoFor(m);

    // Already on disk → treat as "switch to this".
    if (info.state == DownloadState.installed ||
        _installedLlmModels.contains(m)) {
      widget.service.setActiveLlmModel(m);
      _updateDownload(m,
          state: DownloadState.installed, progress: 1.0, clearError: true);
      return;
    }

    // Already actively downloading — guard against double-start. The new
    // CancelToken would orphan the in-flight one.
    if (info.state == DownloadState.downloading) return;

    final estimator = EtaEstimator();
    _etaEstimators[m] = estimator;
    final token = CancelToken();
    _updateDownload(
      m,
      state: DownloadState.downloading,
      progress: info.progress, // preserve resume point if coming from paused
      // etaMin starts at 0 — the UI renders that as "Calculating…" until
      // the estimator has seen enough real progress to produce a number.
      // Quoting a hardcoded baseline (the old behavior) just lied for the
      // first ~3 seconds in either direction.
      etaMin: 0,
      cancelToken: token,
      clearError: true,
    );

    // Keep the engine pointed at whatever the user picked most recently so
    // warm-up post-install hits the right variant. (Switching is cheap if
    // it's already current.)
    if (_selectedModel != null &&
        LlmModel.fromId(_selectedModel!.id) == m) {
      _downloadEta = 0;
    }
    widget.service.setActiveLlmModel(m);

    widget.service
        .installModel(
      model: m,
      token: _huggingFaceToken.isEmpty ? null : _huggingFaceToken,
      cancelToken: token,
      onProgress: (progress) {
        if (!mounted) return;
        // Real ETA: ask the per-download estimator. It blends an EMA
        // of observed progress/sec with an elapsed-based projection
        // during the warmup window so the number reacts to actual
        // network speed instead of just walking down a hardcoded line.
        final newEta = estimator.update(progress);
        _updateDownload(m, progress: progress, etaMin: newEta);
        // Keep the legacy `_downloadEta` in sync with the foregrounded
        // download so the onboarding ModelDownloadScreen still shows
        // accurate "X min left" text.
        if (LlmModel.fromId(_selectedModel?.id ?? '') == m) {
          if (mounted) setState(() => _downloadEta = newEta);
        }
      },
    )
        .then((_) {
      _etaEstimators.remove(m);
      if (!mounted) return;
      _updateDownload(
        m,
        state: DownloadState.installed,
        progress: 1.0,
        clearCancelToken: true,
        clearError: true,
      );
      setState(() => _modelReady = true);
      Timer(const Duration(seconds: 4), () {
        if (mounted) setState(() => _modelReady = false);
      });
      _refreshInstalledLlmModels();
      _kickOffWarmUpIfReady();
    }).catchError((e) {
      _etaEstimators.remove(m);
      if (!mounted) return;
      if (CancelToken.isCancel(e)) {
        // User-initiated pause/cancel. The paused-vs-cancelled distinction
        // is decided when the token is cancelled (the reason field).
        final current = _infoFor(m);
        if (current.state == DownloadState.downloading) {
          // We were still in the "running" state — caller wanted a hard
          // cancel without setting paused first. Reset to idle.
          _updateDownload(m,
              state: DownloadState.idle,
              progress: 0,
              clearCancelToken: true);
        }
      } else {
        debugPrint('[ash] Download error for $m: $e');
        _updateDownload(
          m,
          state: DownloadState.failed,
          errorMessage: e.toString(),
          clearCancelToken: true,
        );
      }
    });
  }

  /// Cancel an in-flight install — discards the partial file (the SDK
  /// can't reliably resume HuggingFace downloads anyway), and resets the
  /// row to idle so the user can re-start cleanly. Not the same as
  /// pause; we don't expose pause at all because the underlying SDK
  /// disables resume for HF URLs.
  Future<void> _cancelDownloadFor(LlmModel m) async {
    final info = _infoFor(m);
    if (info.state == DownloadState.downloading) {
      info.cancelToken?.cancel('cancelled');
    }
    _etaEstimators.remove(m);
    // Best-effort partial cleanup — the SDK doesn't expose a partial path,
    // so we issue a deleteModel which is a no-op when nothing's on disk
    // but cleans up if the partial was renamed to final.
    try {
      await widget.service.deleteModel(model: m);
    } catch (_) {/* ignore */}
    if (!mounted) return;
    _updateDownload(m,
        state: DownloadState.idle,
        progress: 0,
        clearCancelToken: true,
        clearError: true);
    await _refreshInstalledLlmModels();
  }

  // ── Chat interaction ─────────────────────────────────────────

  void _handleSendInActiveChat(String text) {
    if (_isGenerating) return;
    final chat = _activeChat;
    if (chat == null) return;
    if (!_modelOnDisk) {
      _showWaitingForModel();
      return;
    }
    final imageBytes = _pendingImageBytes;
    final prompt = text.trim().isEmpty && imageBytes != null
        ? 'What do you see in this image?'
        : text;
    setState(() {
      chat.messages.add(
        ChatMessage(
          id: chat.messages.length + 1,
          role: 'user',
          text: prompt,
          imageBytes: imageBytes,
        ),
      );
      _pendingImageBytes = null;
    });
    _runInference(chat, prompt, imageBytes: imageBytes);
  }

  void _clearPendingImage() {
    setState(() => _pendingImageBytes = null);
  }

  void _handleNewChat({Set<String>? lensPackIds, String? initialPrompt}) {
    final id = 'c${_nextChatId()}';
    final chat = Chat(
      id: id,
      title: 'New conversation',
      when: 'Now',
      preview: '',
      lensPackIds: lensPackIds,
      initialComposerText: initialPrompt,
    );
    setState(() {
      _chats = [chat, ..._chats];
      _chatId = id;
    });
    // Sync the service to the new chat's lens up front so retrieval is
    // scoped correctly the very first time the user sends a message.
    widget.service.setLens(lensPackIds);
    // Engine may have been closed after the user left a previous chat — make
    // sure the text-only engine is warming before the user types.
    _kickOffWarmUpIfReady();
  }

  /// Snapshot of [chat.messages] for replay into the model when a fresh
  /// session is needed. Drops the just-added user prompt and the empty
  /// streaming placeholder so we don't replay the live turn into itself.
  List<HistoryTurn> _historySnapshotFor(Chat chat) {
    final msgs = chat.messages;
    if (msgs.length < 2) return const [];
    // The last two entries are always (user prompt for current turn,
    // empty assistant placeholder) — exclude both.
    final upTo = msgs.length - 2;
    return [
      for (var i = 0; i < upTo; i++)
        HistoryTurn(
          isUser: msgs[i].role == 'user',
          text: msgs[i].text,
          imageBytes: msgs[i].imageBytes,
        ),
    ];
  }

  /// Eagerly apply [chat.inferenceSettings] to the model so the user doesn't
  /// have to wait when they send their next message. Skips if generation is
  /// in flight — the in-flight stream is still using the old session, and
  /// applying mid-stream would corrupt it. In that case the deferred (per-
  /// query) apply path takes over.
  /// Drop the oldest ~30% of [chat]'s messages and reset the inference
  /// chat session so the next query rebuilds against the trimmed
  /// history. Returns the number of messages dropped (0 means there
  /// weren't enough turns to trim safely). The chat screen reads this
  /// to show its own toast.
  Future<int> _handleContextTrim(Chat chat) async {
    final dropped = trimOldestMessages(chat);
    if (dropped == 0) return 0;
    setState(() {}); // chat is mutated in place — force a rebuild
    try {
      await widget.service.resetChatSession();
    } catch (e) {
      debugPrint('[ash] trim: resetChatSession failed: $e');
    }
    return dropped;
  }

  /// Bump the chat's maxTokens by 2048, capped at the slider's max
  /// (32k). Triggers the same eager engine reload as a settings-sheet
  /// commit so the next message uses the larger window. Returns the
  /// new maxTokens value (== current if already at the cap).
  Future<int> _handleContextExtend(Chat chat) async {
    final current = chat.inferenceSettings.maxTokens;
    if (current >= 32000) return current;
    final next = (current + 2048).clamp(2000, 32000);
    chat.inferenceSettings =
        chat.inferenceSettings.copyWith(maxTokens: next);
    setState(() {});
    await _applySettingsEagerly(chat);
    return next;
  }

  Future<void> _applySettingsEagerly(Chat chat) async {
    if (_isGenerating) return;
    if (!widget.service.isWarm) return; // nothing resident yet — skip
    final needs = widget.service.reconfigureNeedsFor(
      chatId: chat.id,
      settings: chat.inferenceSettings,
      needsVision: false,
    );
    if (!needs.isAnything) return;
    final status = needs.engineReload
        ? _ReconfigureStatus.engineReload
        : _ReconfigureStatus.sessionRebuild;
    setState(() => _reconfigureStatus = status);
    // Session rebuilds can finish in <100ms — the banner would flash and
    // vanish before the user perceives it. Hold the banner for at least
    // this long so the user always gets visible feedback that something
    // ran. Engine reloads take ~5s naturally, so this floor doesn't bite.
    final minVisible = Future<void>.delayed(const Duration(milliseconds: 1500));
    try {
      await Future.wait([
        widget.service.applySettings(
          chatId: chat.id,
          settings: chat.inferenceSettings,
          priorHistory: _historySnapshotForEager(chat),
        ),
        minVisible,
      ]);
    } catch (e, st) {
      debugPrint('[ash] applySettings failed: $e\n$st');
    } finally {
      if (mounted) setState(() => _reconfigureStatus = null);
    }
  }

  /// Like [_historySnapshotFor] but for the eager-apply path: there's no
  /// pending user-turn/assistant-placeholder to exclude, so we replay every
  /// settled message in the chat.
  List<HistoryTurn> _historySnapshotForEager(Chat chat) => [
        for (final m in chat.messages)
          if (!m.streaming)
            HistoryTurn(
              isUser: m.role == 'user',
              text: m.text,
              imageBytes: m.imageBytes,
            ),
      ];

  Future<void> _runInference(Chat chat, String prompt,
      {Uint8List? imageBytes}) async {
    if (_isGenerating) return;
    // Ask the service what reconfiguration this query will force. Pick the
    // single most-informative status — visionLoad and engineReload both imply
    // a long wait; sessionRebuild is the cheap case ("history replay").
    final needs = widget.service.reconfigureNeedsFor(
      chatId: chat.id,
      settings: chat.inferenceSettings,
      needsVision: imageBytes != null,
    );
    final status = needs.visionLoad
        ? _ReconfigureStatus.visionLoad
        : needs.engineReload
            ? _ReconfigureStatus.engineReload
            : needs.sessionRebuild
                ? _ReconfigureStatus.sessionRebuild
                : null;
    setState(() {
      _isGenerating = true;
      _reconfigureStatus = status;
      chat.messages.add(
        ChatMessage(
          id: chat.messages.length + 1,
          role: 'assistant',
          text: '',
          streaming: true,
        ),
      );
    });

    final msgIndex = chat.messages.length - 1;
    var accumulated = '';
    // Captured from the preflight chunk the service emits before token
    // streaming starts. Threaded into every subsequent flush so the citation
    // chips appear as soon as we know what grounded the reply, not only
    // after the stream ends.
    List<MessageSource>? messageSources;

    try {
      // Build replay history: every message in this chat EXCEPT the latest
      // user prompt we're about to send (already appended) and the
      // streaming-placeholder assistant message just added (still empty).
      // The service uses this only when creating a fresh session — same-chat
      // continuations rely on the persisted native chat.
      final replay = _historySnapshotFor(chat);
      // Throttle UI updates to ~30 Hz. On fast streams Gemma 4 emits tokens
      // faster than the framework can re-parse markdown + re-layout the list,
      // causing visible ghosting / dropped frames. We accumulate text in a
      // local buffer and only setState when at least [flushIntervalMs] has
      // passed since the last commit (or when the stream completes).
      const flushIntervalMs = 33;
      var lastFlushMs = DateTime.now().millisecondsSinceEpoch;
      var pendingFlush = false;

      void flush({required bool done}) {
        if (!mounted) return;
        setState(() {
          chat.messages[msgIndex] = ChatMessage(
            id: chat.messages[msgIndex].id,
            role: 'assistant',
            text: accumulated,
            streaming: !done,
            sources: messageSources,
          );
          if (done) {
            chat.preview = accumulated.length > 50
                ? '${accumulated.substring(0, 50)}...'
                : accumulated;
            _isGenerating = false;
            _reconfigureStatus = null;
          } else if (_reconfigureStatus != null) {
            // First visible token → drop any reconfigure banner.
            _reconfigureStatus = null;
          }
        });
        lastFlushMs = DateTime.now().millisecondsSinceEpoch;
        pendingFlush = false;
      }

      await for (final chunk in widget.service.query(
        chatId: chat.id,
        prompt: prompt,
        imageBytes: imageBytes,
        priorHistory: replay,
        settings: chat.inferenceSettings,
      )) {
        if (!mounted) break;

        // Preflight: sources arrive before any tokens. Stash them and flush
        // immediately so the chips render up front.
        if (chunk.sources != null) {
          messageSources = chunk.sources;
          flush(done: false);
          continue;
        }

        if (chunk.isDone) {
          flush(done: true);
          break;
        }

        accumulated += chunk.text;
        final nowMs = DateTime.now().millisecondsSinceEpoch;
        if (nowMs - lastFlushMs >= flushIntervalMs) {
          flush(done: false);
        } else if (!pendingFlush) {
          // Schedule a trailing flush so a brief flurry-then-pause still
          // ends up rendered. Cheap microtask-style timer.
          pendingFlush = true;
          Future.delayed(
            Duration(milliseconds: flushIntervalMs - (nowMs - lastFlushMs)),
            () {
              if (pendingFlush && mounted && _isGenerating) flush(done: false);
            },
          );
        }
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          chat.messages[msgIndex] = ChatMessage(
            id: chat.messages[msgIndex].id,
            role: 'assistant',
            text: accumulated.isEmpty
                ? 'Something went wrong. Try again.'
                : accumulated,
            streaming: false,
            sources: messageSources,
          );
          _isGenerating = false;
          _reconfigureStatus = null;
        });
      }
    }
  }

  void _handleOpenChat(String id) {
    setState(() => _chatId = id);
    // Sync the retrieval lens to whatever this chat was pinned to (or null
    // for "all installed"). Without this, opening a chat after one with a
    // scoped lens would keep the previous scope.
    final chat = _chats.where((c) => c.id == id).firstOrNull;
    widget.service.setLens(chat?.lensPackIds);
    // Mirrors Google AI Edge Gallery: engine lives only while a chat is open.
    // Warm text-only on entry if it isn't already resident.
    _kickOffWarmUpIfReady();
  }

  /// Push the Library reader screen for [packId]. The reader's "Ask Ash"
  /// CTAs route back here through [_askAboutPack] / [_askAboutSection] so
  /// the new chat lands with the right lens (single pack) and an optional
  /// seeded user message.
  ///
  /// When [focusChunkId] is non-empty, the reader scrolls to and highlights
  /// that proposition card on first frame — used by the citation chip's
  /// "Open in Library" deep-link so the user lands on the exact chunk that
  /// grounded the reply, not at the top of the pack.
  void _openPackReader(String packId, [String focusChunkId = '']) {
    final ctx = _navigatorKey.currentContext;
    if (ctx == null) return;
    final pack = _packs.where((p) => p.id == packId).firstOrNull;
    if (pack == null) return;
    Navigator.of(ctx).push(
      MaterialPageRoute<void>(
        builder: (_) => PackReaderScreen(
          pack: pack,
          service: widget.service,
          focusChunkId: focusChunkId.isEmpty ? null : focusChunkId,
          onAskAboutPack: () {
            // Pop the reader so the new chat takes the foreground.
            Navigator.of(ctx).pop();
            _askAboutPack(pack.id);
          },
          onAskAboutSection: (sectionLabel) {
            Navigator.of(ctx).pop();
            _askAboutSection(pack.id, sectionLabel);
          },
        ),
      ),
    );
  }

  void _askAboutPack(String packId) {
    _handleNewChat(lensPackIds: {packId});
  }

  void _askAboutSection(String packId, String seedMessage) {
    _handleNewChat(lensPackIds: {packId});
    // The seed IS the user's first message — for a Library proposition,
    // that's the proposition's own question (which will retrieve back the
    // same chunk for high-confidence grounding). For the search-bar
    // "Ask Ash this" CTA, it's whatever the user typed. Either way it
    // lands verbatim in the chat as the user's turn.
    //
    // If the model isn't on disk yet, prefill the seed into the composer
    // instead of trying to auto-send — the user can tap send themselves
    // once download lands. _handleSendInActiveChat would otherwise toast
    // and drop the seed, leaving the chat empty.
    if (!_modelOnDisk) {
      final chat = _activeChat;
      if (chat != null) {
        setState(() => chat.initialComposerText = seedMessage);
      }
      _showWaitingForModel();
      return;
    }
    Future.microtask(() {
      if (!mounted) return;
      _handleSendInActiveChat(seedMessage);
    });
  }

  /// Semantic ranking for the Store search box. Embeds every known pack
  /// (name + summary) via the retrieval-time MiniLM, then cosine-ranks
  /// against the user's query. Cheap after the first call — embeddings are
  /// cached service-side.
  Future<List<PackRanking>> _rankPacksForStore(String query) {
    final texts = <String, String>{
      for (final p in _packs) p.id: '${p.name}\n${p.summary}',
    };
    return widget.service.rankByQuery(query: query, texts: texts);
  }

  /// Install [pack] — dispatches to the bundle path or the remote CDN path
  /// based on [Pack.source]. The progress callback bubbles real bytes-read
  /// updates (for remote) or pulses to 1.0 once the chunks are committed
  /// (for bundle, which completes in tens of milliseconds).
  Future<void> _handleInstallPack(
    Pack pack,
    void Function(double progress)? onProgress,
  ) async {
    if (pack.source == PackSource.remote && pack.downloadUrl != null) {
      await widget.service.importPackFromUrl(
        pack.id,
        pack.downloadUrl!,
        onProgress: onProgress,
      );
    } else {
      await widget.service.importPack(pack.id);
      onProgress?.call(1.0);
    }
  }

  /// Replace the active chat's lens (a Set of packIds, or null = all
  /// installed). Fired from the lens-pill bottom sheet in the chat header.
  void _handleChangeLens(Set<String>? lensPackIds) {
    final chat = _activeChat;
    if (chat == null) return;
    setState(() => chat.lensPackIds = lensPackIds);
    widget.service.setLens(lensPackIds);
  }

  void _handleCloseChat() {
    // The composer TextField holds focus while typing — without an explicit
    // unfocus here iOS keeps the keyboard mounted after we rebuild without
    // the chat screen, leaving it floating over the Home tab.
    FocusManager.instance.primaryFocus?.unfocus();
    setState(() {
      _chatId = null;
      _reconfigureStatus = null;
    });
    // Drop the chat KV but keep the engine warm. LiteRT-LM has a state-leak
    // bug where destroying an engine that has served a vision query corrupts
    // the runtime, and every subsequent `litert_lm_engine_create` in the same
    // process returns null in ~200ms regardless of backend. Skipping the
    // engine destroy avoids the trap and also makes chat switching instant.
    widget.service.resetChatSession();
  }

  void _handleDeleteChat(String id) {
    final wasActive = _chatId == id;
    setState(() {
      _chats = _chats.where((c) => c.id != id).toList();
      if (wasActive) _chatId = null;
    });
    if (wasActive) widget.service.resetChatSession();
  }

  void _handleRenameChat(String id, String newTitle) {
    final trimmed = newTitle.trim();
    if (trimmed.isEmpty) return;
    final chat = _chats.where((c) => c.id == id).firstOrNull;
    if (chat == null) return;
    setState(() => chat.title = trimmed);
  }

  void _handleToggleStarChat(String id) {
    final chat = _chats.where((c) => c.id == id).firstOrNull;
    if (chat == null) return;
    setState(() => chat.starred = !chat.starred);
  }

  // ── Live voice mode plumbing ─────────────────────────────────

  /// Ensures there's an active Chat when the user enters live voice mode
  /// from a screen where none is open (home tab). Creates a fresh chat and
  /// keeps subsequent live-mode turns inside it.
  Chat _ensureLiveChat() {
    final existing = _activeChat;
    if (existing != null) return existing;
    final id = 'c${_nextChatId()}';
    final chat = Chat(
      id: id,
      title: 'Live conversation',
      when: 'Now',
      preview: '',
      // Default lens: all installed packs.
    );
    _chats = [chat, ..._chats];
    _chatId = id;
    widget.service.setLens(chat.lensPackIds);
    return chat;
  }

  /// Snapshot of chat history for live-mode replay. Unlike the chat-screen
  /// helper, the messages list here does NOT yet contain the current user
  /// turn (live mode adds the turn AFTER receiving the final transcript),
  /// so we don't trim it.
  List<HistoryTurn> _historySnapshotForLive(Chat chat) {
    return [
      for (final m in chat.messages)
        HistoryTurn(
          isUser: m.role == 'user',
          text: m.text,
          imageBytes: m.imageBytes,
        ),
    ];
  }

  void _handleLiveUserTurn(String text) {
    final chat = _activeChat;
    if (chat == null) return;
    setState(() {
      chat.messages.add(
        ChatMessage(
          id: chat.messages.length + 1,
          role: 'user',
          text: text,
        ),
      );
      chat.preview = text.length > 50 ? '${text.substring(0, 50)}...' : text;
      if (chat.title == 'Live conversation' ||
          chat.title == 'New conversation') {
        chat.title = text.length > 40 ? '${text.substring(0, 40)}...' : text;
      }
    });
  }

  void _handleLiveAssistantTurn(String text, List<MessageSource>? sources) {
    final chat = _activeChat;
    if (chat == null) return;
    setState(() {
      chat.messages.add(
        ChatMessage(
          id: chat.messages.length + 1,
          role: 'assistant',
          text: text,
          // Preserve RAG provenance so the chip row renders when the
          // user returns to the text chat. Null when RAG was off or the
          // query was image-only (no preflight sources chunk).
          sources: sources,
        ),
      );
      chat.preview = text.length > 50 ? '${text.substring(0, 50)}...' : text;
    });
  }

  Future<void> _handlePickAttachment() async {
    if (!_modelOnDisk) {
      _showWaitingForModel(
          'Ash is still downloading. You can attach a photo once setup finishes.');
      return;
    }
    final navContext = _navigatorKey.currentContext;
    if (navContext == null) return;
    final source = await showModalBottomSheet<ImageSource>(
      context: navContext,
      backgroundColor: Colors.transparent,
      builder: (sheetContext) => _ImageSourceSheet(
        onPick: (s) => Navigator.of(sheetContext).pop(s),
      ),
    );
    if (source == null) return;

    final picker = ImagePicker();
    try {
      final image = await picker.pickImage(
        source: source,
        maxWidth: 512,
        maxHeight: 512,
        imageQuality: 80,
      );
      if (image == null) return; // User cancelled
      final bytes = await image.readAsBytes();
      if (!mounted) return;
      setState(() => _pendingImageBytes = bytes);
    } catch (e) {
      debugPrint('[ash] Image pick failed: $e');
    }
  }

  // ── Build ────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      navigatorKey: _navigatorKey,
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: AshColors.dark.bg,
        extensions: const [AshColors.dark],
        colorScheme: const ColorScheme.dark(
          surface: Color(0xFF0F0F0F),
          primary: Color(0xFFCC3F1E),
        ),
      ),
      home: _buildBody(),
    );
  }

  Widget _buildBody() {
    return Stack(
      children: [
        _buildBackground(),
        switch (_stage) {
          AppStage.onboarding => OnboardingScreen(
              onDone: () {
                _markOnboardingDone();
                _setStage(AppStage.modelPick);
              },
            ),
          AppStage.modelPick => ModelPickScreen(
              onSelect: (model) {
                setState(() {
                  _selectedModel = model;
                  _tab = MainTab.models;
                });
                _startDownload();
                // Skip the dedicated full-screen "downloading" stage — the
                // Models tab already shows the in-flight row with progress,
                // and the floating banner surfaces it across other tabs.
                // One fewer ceremony screen the user has to dismiss.
                _setStage(AppStage.main);
              },
            ),
          AppStage.downloading => ModelDownloadScreen(
              model: _selectedModel!,
              progress: _selectedDownloadProgress,
              etaMin: _downloadEta,
              onContinue: () => _setStage(AppStage.main),
              // Legacy: still routable from old code paths (e.g. tapping the
              // floating banner before we switched it to route to the Models
              // tab). Kept for compatibility — new flows skip this entirely.
              onBack: _installedLlmModels.isEmpty
                  ? null
                  : () => _setStage(AppStage.main),
            ),
          AppStage.main => _buildMainStage(),
        },
      ],
    );
  }

  // ── Background ───────────────────────────────────────────────

  Widget _buildBackground() {
    return Positioned.fill(
      child: Container(
        color: AshColors.dark.bg,
        child: Stack(
          children: [
            Positioned(
              left: -80,
              top: -80,
              child: Container(
                width: 320,
                height: 320,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  gradient: RadialGradient(
                    colors: [
                      AshColors.dark.accentGlow,
                      AshColors.dark.accentGlow.withValues(alpha: 0),
                    ],
                  ),
                ),
              ),
            ),
            Positioned(
              right: -100,
              bottom: -60,
              child: Container(
                width: 280,
                height: 280,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  gradient: RadialGradient(
                    colors: [
                      AshColors.dark.accentSoft,
                      AshColors.dark.accentSoft.withValues(alpha: 0),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ── Main stage ───────────────────────────────────────────────

  Widget _buildMainStage() {
    final activeChat = _activeChat;
    final installed = _installedPacks;

    // Show loading while packs are being loaded.
    if (_packs.isEmpty) {
      return const Material(
        color: Colors.transparent,
        child: Center(child: CircularProgressIndicator()),
      );
    }

    return Material(
      color: Colors.transparent,
      child: Stack(
        children: [
          Column(
            children: [
              Expanded(
                child: activeChat != null
                    ? ChatScreen(
                        chat: activeChat,
                        installedPacks: installed,
                        modelReady: _modelOnDisk,
                        activeLlmModel: widget.service.activeLlmModel,
                        onOpenModelPicker: () =>
                            setState(() => _sheet = 'settings'),
                        onBack: _handleCloseChat,
                        onOpenVoice: () {
                          if (!_modelOnDisk) {
                            _showWaitingForModel(
                                'Live voice opens once Ash finishes downloading.');
                            return;
                          }
                          setState(() => _sheet = 'voice');
                        },
                        onOpenCamera: _handlePickAttachment,
                        onSendInChat: _handleSendInActiveChat,
                        onSettingsChanged: (_) => setState(() {}),
                        onSettingsCommitted: () =>
                            _applySettingsEagerly(activeChat),
                        onRename: (newTitle) =>
                            _handleRenameChat(activeChat.id, newTitle),
                        onChangeLens: _handleChangeLens,
                        onOpenInLibrary: _openPackReader,
                        onContextTrim: () => _handleContextTrim(activeChat),
                        onContextExtend: () =>
                            _handleContextExtend(activeChat),
                        pendingImageBytes: _pendingImageBytes,
                        onClearPendingImage: _clearPendingImage,
                        voiceService: widget.voiceService,
                      )
                    : switch (_tab) {
                        MainTab.home => HomeScreen(
                            chats: _chats,
                            onOpenChat: _handleOpenChat,
                            onNew: () => _handleNewChat(),
                            onNewWithPrompt: (prompt) =>
                                _handleNewChat(initialPrompt: prompt),
                            onDeleteChat: _handleDeleteChat,
                            onRenameChat: _handleRenameChat,
                            onToggleStarChat: _handleToggleStarChat,
                          ),
                        MainTab.knowledge => KnowledgeScreen(
                            packs: _packs,
                            onPacksChanged: (p) => setState(() => _packs = p),
                            onAddPack: _handleInstallPack,
                            onRemovePack: (packId) =>
                                widget.service.uninstallPack(packId),
                            onOpenPack: _openPackReader,
                            onRankPacks: _rankPacksForStore,
                          ),
                        MainTab.models => ModelsScreen(
                            activeLlmModel: widget.service.activeLlmModel,
                            installedLlmModels: _installedLlmModels,
                            downloads: _downloads,
                            onDownload: _startDownloadFor,
                            onCancel: (m) => _cancelDownloadFor(m),
                            onSwitch: _switchActiveLlmModel,
                            onDelete: _deleteLlmModel,
                          ),
                        MainTab.profile => ProfileScreen(
                            onOpenSettings: () =>
                                setState(() => _sheet = 'settings'),
                            activeLlmModel: widget.service.activeLlmModel,
                            installedLlmModels: _installedLlmModels,
                            onSwitchLlmModel: _switchActiveLlmModel,
                            onInstallLlmModel: (m) {
                              setState(() => _selectedModel =
                                  _findModelInfoFor(m));
                              _setStage(AppStage.downloading);
                              _startDownload();
                            },
                            onDeleteLlmModel: _deleteLlmModel,
                            packsInstalled:
                                _packs.where((p) => p.installed).length,
                            chatsCount: _chats.length,
                            // Models report sizeGB in gigabytes; packs report
                            // bytes as raw bytes. Convert pack bytes → GB
                            // before summing so we don't end up with absurd
                            // 8M GB totals.
                            storageBytesUsed:
                                _installedLlmModels.fold<double>(
                                      0.0,
                                      (sum, m) => sum + m.sizeGB,
                                    ) +
                                    _packs
                                        .where((p) => p.installed)
                                        .fold<double>(
                                            0.0,
                                            (sum, p) =>
                                                sum +
                                                p.bytes /
                                                    (1024 * 1024 * 1024)),
                          ),
                      },
              ),
              if (activeChat == null)
                Padding(
                  padding: const EdgeInsets.only(bottom: 16),
                  child: ash.AshTabBar(
                    activeTab: _tab,
                    onChanged: (t) {
                      // Drop focus on tab switch — otherwise a focused search
                      // field on one tab leaves the keyboard up across tabs.
                      FocusManager.instance.primaryFocus?.unfocus();
                      setState(() => _tab = t);
                    },
                  ),
                ),
            ],
          ),

          if (_sheet == 'voice')
            Positioned.fill(
              child: LiveVoiceScreen(
                chat: _activeChat ?? _ensureLiveChat(),
                service: widget.service,
                voiceService: widget.voiceService,
                priorHistorySnapshot: () => _activeChat == null
                    ? const <HistoryTurn>[]
                    : _historySnapshotForLive(_activeChat!),
                onUserTurn: _handleLiveUserTurn,
                onAssistantTurn: _handleLiveAssistantTurn,
                onExit: () => setState(() => _sheet = null),
                onContextExtend: () =>
                    _handleContextExtend(_activeChat ?? _ensureLiveChat()),
                onContextTrim: () =>
                    _handleContextTrim(_activeChat ?? _ensureLiveChat()),
              ),
            ),

          if (_sheet == 'settings')
            Positioned.fill(
              child: SettingsScreen(
                onBack: () => setState(() => _sheet = null),
                accelerator: widget.service.accelerator,
                onAcceleratorChanged: (choice) async {
                  await widget.service.setAccelerator(choice);
                  if (!mounted) return;
                  setState(() {});
                  // Engine was closed by setAccelerator; re-warm text-only so
                  // the next message doesn't pay a ~5s cold load on top of
                  // whatever else the user does.
                  _kickOffWarmUpIfReady();
                },
                activeLlmModel: widget.service.activeLlmModel,
                installedLlmModels: _installedLlmModels,
                onSwitchLlmModel: (m) {
                  setState(() => _sheet = null);
                  _switchActiveLlmModel(m);
                },
                onInstallLlmModel: (m) {
                  setState(() {
                    _sheet = null;
                    _selectedModel = _findModelInfoFor(m);
                  });
                  _setStage(AppStage.downloading);
                  _startDownload();
                },
                onDeleteLlmModel: _deleteLlmModel,
                onClearConversations: _clearAllConversations,
                voiceService: widget.voiceService,
                speculativeDecoding:
                    widget.service.speculativeDecoding,
                onSpeculativeDecodingChanged: _toggleSpeculativeDecoding,
              ),
            ),

          // Download banner — shown in main stage while ANY install is in
          // progress (regardless of which variant is foregrounded). Floats
          // above the tab bar. Tapping returns to the dedicated download
          // screen for the foregrounded variant.
          if (_stage != AppStage.downloading && _isAnyDownloading)
            Positioned(
              left: 16,
              right: 16,
              bottom: 100 + MediaQuery.of(context).viewInsets.bottom,
              child: GestureDetector(
                behavior: HitTestBehavior.opaque,
                // Tap routes to the Models tab — the canonical place to
                // see per-variant progress, cancel, switch active, etc.
                // The old full-screen download view is legacy.
                onTap: () {
                  if (_activeChat != null) _handleCloseChat();
                  setState(() => _tab = MainTab.models);
                },
                child: _DownloadBanner(),
              ),
            ),

          if (_modelReady)
            Positioned(
              top: 60,
              left: 16,
              right: 16,
              child: _ModelReadyToast(
                  onClose: () => setState(() => _modelReady = false)),
            ),

          if (_isWarmingUp && !_modelReady)
            const Positioned(
              top: 60,
              left: 16,
              right: 16,
              child: _WarmUpBanner(),
            ),

          if (_waitingForModelToast != null)
            Positioned(
              top: 60,
              left: 16,
              right: 16,
              child: _WaitingForModelToast(message: _waitingForModelToast!),
            ),

          if (_reconfigureStatus != null)
            Positioned(
              top: 60,
              left: 16,
              right: 16,
              child: _ReconfigureBanner(status: _reconfigureStatus!),
            ),
        ],
      ),
    );
  }
}

// ─── Image source picker sheet ──────────────────────────────────────────────

class _ImageSourceSheet extends StatelessWidget {
  const _ImageSourceSheet({required this.onPick});

  final ValueChanged<ImageSource> onPick;

  @override
  Widget build(BuildContext context) {
    final c = AshColors.dark;
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
        child: Glass(
          radius: 24,
          strong: true,
          padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 16),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 36,
                height: 4,
                decoration: BoxDecoration(
                  color: c.textDim,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
              const SizedBox(height: 16),
              Text(
                'Add a photo',
                style: TextStyle(
                  color: c.text,
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(height: 14),
              _SourceTile(
                icon: Icons.camera_alt_outlined,
                label: 'Take photo',
                onTap: () => onPick(ImageSource.camera),
                c: c,
              ),
              const SizedBox(height: 8),
              _SourceTile(
                icon: Icons.photo_library_outlined,
                label: 'Choose from album',
                onTap: () => onPick(ImageSource.gallery),
                c: c,
              ),
              const SizedBox(height: 12),
              GestureDetector(
                onTap: () => Navigator.of(context).pop(),
                child: Container(
                  height: 44,
                  alignment: Alignment.center,
                  child: Text(
                    'Cancel',
                    style: TextStyle(
                      color: c.textMuted,
                      fontSize: 14,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _SourceTile extends StatelessWidget {
  const _SourceTile({
    required this.icon,
    required this.label,
    required this.onTap,
    required this.c,
  });

  final IconData icon;
  final String label;
  final VoidCallback onTap;
  final AshColors c;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        height: 52,
        padding: const EdgeInsets.symmetric(horizontal: 16),
        decoration: BoxDecoration(
          color: c.surfaceElev,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: c.border, width: 0.5),
        ),
        child: Row(
          children: [
            Icon(icon, color: c.accent, size: 22),
            const SizedBox(width: 14),
            Text(
              label,
              style: TextStyle(
                color: c.text,
                fontSize: 15,
                fontWeight: FontWeight.w500,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ─── Download banner ─────────────────────────────────────────────────────────

class _DownloadBanner extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final appState = context.findAncestorStateOfType<_AshAppState>();
    final c = AshColors.dark;

    // Bind the banner to the variant that's actually downloading right
    // now, NOT to `_selectedModel` (the user's last-picked variant). The
    // two diverge in two real cases:
    //   1. User picked E2B, it installed, then they tapped Download on
    //      E4B in the Models tab — `_selectedModel` still says E2B
    //      (100% installed), so the banner would show "Downloading E2B
    //      · 100%" while E4B was the actual download.
    //   2. User switched active to an installed variant while a
    //      different variant was downloading — the banner used to snap
    //      to the just-switched model's progress (0% since it wasn't
    //      downloading), showing the misleading
    //      "Downloading · Calculating · 0%" state the user reported.
    // The downloading variant is the only thing the banner should ever
    // care about — pull all its data from there.
    final downloading = appState?._downloadingVariant;
    final info = downloading == null
        ? const ModelDownloadInfo()
        : appState!._infoFor(downloading);
    final progress = info.progress;
    final etaMin = info.etaMin;
    return Glass(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      radius: 12,
      blur: 20,
      child: Row(
        children: [
          ProgressRing(
            progress: progress,
            size: 32,
            strokeWidth: 2.5,
            child: Text(
              '${(progress * 100).round()}%',
              style: TextStyle(
                color: c.textDim,
                fontSize: 10,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  downloading == null
                      ? 'Downloading'
                      : 'Downloading ${downloading.displayName}',
                  style: TextStyle(
                    color: c.text,
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                Text(
                  etaMin > 0
                      ? '~$etaMin min remaining'
                      : 'Calculating…',
                  style: TextStyle(
                    color: c.textDim,
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ─── Model ready toast ────────────────────────────────────────────────────────

class _ModelReadyToast extends StatelessWidget {
  final VoidCallback onClose;

  const _ModelReadyToast({required this.onClose});

  @override
  Widget build(BuildContext context) {
    final appState = context.findAncestorStateOfType<_AshAppState>();
    final c = AshColors.dark;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: c.surfaceElev,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: c.accent,
          width: 1.5,
        ),
        boxShadow: [
          BoxShadow(
            color: c.accentGlow,
            blurRadius: 20,
            offset: const Offset(0, 0),
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            width: 32,
            height: 32,
            decoration: BoxDecoration(
              color: c.accent,
              shape: BoxShape.circle,
              boxShadow: [
                BoxShadow(
                  color: c.accentGlow,
                  blurRadius: 12,
                  offset: const Offset(0, 0),
                ),
              ],
            ),
            child: Icon(
              Icons.check,
              color: Colors.white,
              size: 18,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '${appState?._selectedModel?.name ?? 'Model'} ${appState?._selectedModel?.param ?? ''} is ready',
                  style: TextStyle(
                    color: c.text,
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  'Ash is now running fully on your device.',
                  style: TextStyle(
                    color: c.textMuted,
                    fontSize: 13,
                  ),
                ),
              ],
            ),
          ),
          IconBtn(
            icon: Icons.close,
            onTap: onClose,
            size: 28,
          ),
        ],
      ),
    );
  }
}

// ─── Waiting-for-model toast ─────────────────────────────────────────────────
// Shown briefly when the user taps a model-gated control (send, camera,
// live voice) while the on-disk model file is still downloading. Echoes
// the bottom-of-screen download banner so the user knows what they're
// waiting on without yelling at them.

class _WaitingForModelToast extends StatelessWidget {
  const _WaitingForModelToast({required this.message});

  final String message;

  @override
  Widget build(BuildContext context) {
    final c = AshColors.dark;
    return Glass(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      radius: 14,
      blur: 20,
      child: Row(
        children: [
          Icon(Icons.hourglass_bottom_rounded, size: 18, color: c.accent),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              message,
              style: TextStyle(
                color: c.text,
                fontSize: 13.5,
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ─── Warm-up banner ───────────────────────────────────────────────────────────

class _WarmUpBanner extends StatelessWidget {
  const _WarmUpBanner();

  @override
  Widget build(BuildContext context) {
    final c = AshColors.dark;
    return Glass(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      radius: 14,
      blur: 20,
      child: Row(
        children: [
          SizedBox(
            width: 18,
            height: 18,
            child: CircularProgressIndicator(
              strokeWidth: 2,
              valueColor: AlwaysStoppedAnimation<Color>(c.accent),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  'Warming up on-device model…',
                  style: TextStyle(
                    color: c.text,
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                Text(
                  'First run can take ~60s. After this, replies are instant.',
                  style: TextStyle(color: c.textMuted, fontSize: 12),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ─── Reconfigure banner ──────────────────────────────────────────────────────
//
// One banner with three modes, each driven by [_ReconfigureStatus]. The
// status is computed up front from [InferenceService.reconfigureNeedsFor]
// when the user sends a message, and cleared as soon as the first token
// arrives.

enum _ReconfigureStatus { visionLoad, engineReload, sessionRebuild }

class _ReconfigureBanner extends StatelessWidget {
  const _ReconfigureBanner({required this.status});

  final _ReconfigureStatus status;

  @override
  Widget build(BuildContext context) {
    final c = AshColors.dark;
    final (title, subtitle) = switch (status) {
      _ReconfigureStatus.visionLoad => (
          'Loading vision capability…',
          'First image of the chat — about 30s. Quicker after.',
        ),
      _ReconfigureStatus.engineReload => (
          'Reloading engine…',
          'Applying new context size. Text ~5s, with vision ~30s.',
        ),
      _ReconfigureStatus.sessionRebuild => (
          'Refreshing chat session…',
          'Applying new settings and replaying history.',
        ),
    };
    return Glass(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      radius: 14,
      blur: 20,
      child: Row(
        children: [
          SizedBox(
            width: 18,
            height: 18,
            child: CircularProgressIndicator(
              strokeWidth: 2,
              valueColor: AlwaysStoppedAnimation<Color>(c.accent),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  title,
                  style: TextStyle(
                    color: c.text,
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                  ),
                ),
                Text(
                  subtitle,
                  style: TextStyle(color: c.textMuted, fontSize: 12),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
