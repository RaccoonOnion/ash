import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_gemma/flutter_gemma.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:path_provider/path_provider.dart';

import 'bert_tokenizer.dart';
import 'bm25_search.dart';
import 'chunk_entity.dart';
import 'chunk_sanitizer.dart';
import 'inference_service.dart';
import 'inference_settings.dart';
import 'llm_model.dart';
import '../objectbox.g.dart';

/// Simple data class for a retrieved RAG chunk.
///
/// [score] is an RRF (Reciprocal Rank Fusion) score after hybrid retrieval —
/// higher is a better match. Before fusion, the vector-only path used cosine
/// distance where lower was better. The RRF score has no fixed upper bound;
/// its magnitude depends on how many ranked lists contribute and the k
/// constant. The list is pre-sorted descending (best first) so that
/// [_assembleContext] can assign [1], [2] markers in order.
class RetrievedInfo {
  const RetrievedInfo({
    required this.chunkId,
    required this.text,
    required this.source,
    required this.sourceType,
    required this.packId,
    required this.packName,
    required this.sectionPath,
    required this.score,
  });

  final String chunkId;
  final String text;
  final String source;
  final String sourceType;
  final String packId;
  final String packName;
  final String? sectionPath;
  final double score;
}

/// Concrete [InferenceService] that combines RAG retrieval with Gemma 4
/// generation. Adapts the FlutterGemmaRagService from gemma4_bench into Ash's
/// simpler InferenceService interface.
class GemmaInferenceService implements InferenceService {
  // --- Constants ---

  /// Default variant used when nothing else is selected — first-run install
  /// and the failsafe when [setActiveLlmModel] hasn't been called.
  static const _defaultLlmModel = LlmModel.gemma4E2B;

  static const _dbSubdirectory = 'rag_core';
  static const _packsDir = 'assets/rag/packs';
  static const _registryAsset = 'assets/rag/packs/packs_registry.json';
  static const _embeddingModelAsset = 'assets/models/minilm.onnx';
  static const _vocabAsset = 'assets/models/vocab.txt';
  static const _embeddingDimensions = 384;
  static const _maxSeqLength = 128;

  // --- RAG tuning ---

  /// Pull this many candidates from HNSW before threshold-filtering. Slightly
  /// over-fetching is cheap (HNSW is microseconds for 384D / ~1k chunks) and
  /// gives the threshold gate something to work with.
  static const _retrieveTopK = 5;

  /// BM25 candidate pool size — pulled independently of vector search and
  /// fused via Reciprocal Rank Fusion.
  static const _bm25CandidateK = 10;

  /// RRF constant k. Higher dampens the contribution of top-ranked items.
  /// 60 is the standard default from Cormack, Clarke & Büttcher (2009).
  static const _rrfK = 60;

  /// Cosine-distance ceiling for a chunk to be considered relevant.
  /// ObjectBox returns the raw cosine distance — lower is better.
  ///
  /// Empirically tuned against round-5 PROPOSITION embeddings (each
  /// chunk's embedding represents its question). MiniLM is small enough
  /// that beyond ~0.7 distance, results are dominated by sentence-
  /// STRUCTURE similarity, not topic — e.g. "i got shot" lands closer
  /// to "What should I do if my car catches fire" (0.74) than to "How
  /// do I treat a wound?" (0.82), because both phrase a What-should-I-
  /// do situation. 0.65 is the boundary where matches are *topically*
  /// confident; beyond that we're guessing.
  ///
  /// Net effect: terse conversational queries ("i got shot", "bleeding",
  /// "fire") need either the matching pack to be installed OR a fuller
  /// query form ("How do I treat a gunshot wound?"). Below the threshold,
  /// the model falls back to general knowledge via the strengthened
  /// system prompt — which is preferable to mis-citing a fire chunk for
  /// a gunshot question.
  static const _maxScoreThreshold = 0.65;

  /// Total character budget for assembled context. Sized to leave plenty of
  /// room in a 4000-token engine for the system prompt + conversation
  /// history + user prompt + assistant reply. ~3000 chars ≈ ~800 tokens —
  /// fits two full propositions with headroom.
  static const _contextCharBudget = 3000;

  // --- State ---

  // Minimum maxTokens for a vision-capable engine. Image patches (~2400 per
  // photo after SDK resize) eat context; below 4096 the FFI throws a
  // DYNAMIC_UPDATE_SLICE prep error or hangs silently. See
  // gemma4_bench/docs/known_runtime_issues.md. Image queries against a chat
  // configured for a smaller window are silently upgraded to this floor.
  static const _visionMinMaxTokens = 4096;
  // Default for pre-warm before any chat is known. Matches
  // InferenceSettings.defaults.maxTokens.
  static const _defaultMaxTokens = 4000;

  Store? _store;
  Box<ChunkEntity>? _box;
  // The InferenceChat is now persisted across turns so the model sees
  // conversation history. It lives until either (a) the user switches
  // chatId, (b) we swap engines (text↔vision), or (c) closeEngine() runs.
  InferenceChat? _persistedChat;
  String? _persistedChatId;
  // Sampling-param signature the persisted chat was created with. When the
  // user changes temp/topK/topP in settings, this no longer matches and the
  // next query rebuilds the chat session.
  String? _persistedChatSamplingSig;
  // Currently resident FFI engine. The user-visible app keeps a text-only
  // engine resident by default (cheap to run, low GPU draw). The engine is
  // swapped to a vision-enabled one the first time an image query arrives
  // in a chat, then stays vision-enabled until [closeEngine] is called.
  InferenceModel? _loadedModel;
  bool _loadedSupportsImage = false;
  // KV-cache size baked into the resident engine. Tracked so we can detect
  // when the user opens a chat whose maxTokens setting differs and rebuild
  // (per Google AI Edge Gallery's pattern: maxTokens IS the engine context
  // window, not a per-response cap).
  int? _loadedMaxTokens;
  // Accelerator currently baked into the resident engine. GPU is the
  // default — Metal is faster and runs cooler than the iOS CPU path. The
  // user can flip this in global Settings to compare.
  AcceleratorChoice _accelerator = AcceleratorChoice.gpu;

  // Set true the first time [closeEngine] runs. Workaround for a hard
  // SIGSEGV in Apple's Metal blit context observed only on TestFlight-
  // signed Release builds: the FIRST engine creation on GPU succeeds,
  // but any close+rebuild path crashes inside
  // `AGX::LegacyBlitContext::copyBufferToBuffer` (+0x68 null deref)
  // when LiteRT-LM's Metal delegate writes constant tensor data into
  // the GPU buffer during `DelegateKernel::Initialize`. Because the
  // SIGSEGV kills the process before any Dart `try/catch` can run, we
  // can't recover dynamically — we have to avoid hitting Metal on the
  // rebuild path in the first place. Once we've closed once, all
  // subsequent engine creates use CPU regardless of [_accelerator].
  // The flag never resets within a session; a relaunch is the only
  // way back to GPU.
  bool _gpuRebuildUnsafe = false;
  // Speculative decoding (Multi-Token Prediction). Engine-level — toggling
  // closes the engine and re-warms with the new flag. Default true because
  // both shipping Gemma 4 variants (E2B-it, E4B-it) carry an MTP drafter
  // and get a real decode speedup from it; non-MTP `.litertlm` files
  // silently ignore the flag at the SDK level.
  bool _speculativeDecoding = true;
  // Currently selected LLM variant. The resident engine is built from this
  // variant's `.litertlm` file. Swapped via [setActiveLlmModel] — closes the
  // engine; the next query re-warms with the new variant.
  LlmModel _activeModel = _defaultLlmModel;
  Completer<void>? _warmUpCompleter;
  Completer<void>? _visionSwapCompleter;
  bool _stopRequested = false;
  // True while a query() stream is actively running. Set at the start
  // of streaming, cleared in the finally. Surfaced via [isGenerating]
  // so the screen tearing down (back-tap during AI reply) can skip
  // calling stopGeneration() on a chat that's already finished — the
  // FFI segfaults on that and native crashes can't be caught in Dart.
  bool _generating = false;
  // Reference to the chat that's currently mid-generation. Set BEFORE
  // addQueryChunk (which triggers prefill) and cleared in the finally.
  // stop() uses this to interrupt prefill OR streaming via a direct
  // chat.stopGeneration() call — the in-loop flag check can only stop
  // generation BETWEEN tokens, so it's useless during prefill (where
  // the await-for hasn't yielded yet) and useless when the consumer
  // cancels the subscription (which aborts the loop before the check
  // runs).
  InferenceChat? _activeChat;
  // Idempotency guard. Both the in-loop stop path and the direct stop()
  // path funnel through this — at most one stopGeneration() call per
  // generation, regardless of who triggered it. Reset at the start of
  // every query().
  bool _stopGenerationCalled = false;
  OnnxRuntime? _ort;
  OrtSession? _embeddingSession;
  BertTokenizer? _tokenizer;

  /// Pack scoping for retrieval. `null` = "search every installed pack"
  /// (the default for new chats). A non-empty set restricts the HNSW query
  /// to chunks whose [ChunkEntity.packId] is in the set. Updated by
  /// [setLens] whenever the active chat or its lens changes.
  Set<String>? _lensPackIds;

  // BM25 sparse-retrieval index. Built lazily on first keyword search and
  // invalidated whenever packs are imported or uninstalled.
  final _bm25 = Bm25Search();
  Future<void>? _bm25BuildFuture;

  // packId → human-readable pack name. Populated from each pack JSON's
  // top-level `packName` field at import time. Used by retrieval to label
  // source blocks like "[1] CPR — During > For Adults" without forcing the
  // caller to thread pack metadata through the service interface.
  final Map<String, String> _packIdToName = {};

  // --- InferenceService interface ---

  @override
  LlmModel get activeLlmModel => _activeModel;

  @override
  Future<void> setActiveLlmModel(LlmModel model) async {
    if (model == _activeModel) return;
    debugPrint('[ash] active model change: ${_activeModel.filename} → '
        '${model.filename} — closing engine; next query will re-warm');
    _activeModel = model;
    await closeEngine();
  }

  @override
  Future<ModelStatus> getModelStatus({LlmModel? model}) async {
    final target = model ?? _activeModel;
    var isInstalled = false;
    String? error;

    try {
      final installedModelIds = await FlutterGemma.listInstalledModels();
      isInstalled = installedModelIds.contains(target.filename);
    } catch (exception) {
      error = 'listInstalledModels failed: $exception';
    }

    if (!isInstalled) {
      try {
        isInstalled = await FlutterGemma.isModelInstalled(target.filename);
      } catch (exception) {
        error = [
          if (error != null) error,
          'isModelInstalled failed: $exception',
        ].join('\n');
      }
    }

    return ModelStatus(isInstalled: isInstalled, error: error);
  }

  @override
  Future<void> installModel({
    LlmModel? model,
    String? token,
    void Function(double progress)? onProgress,
    CancelToken? cancelToken,
  }) async {
    final target = model ?? _activeModel;
    var builder = FlutterGemma.installModel(
      modelType: ModelType.gemma4,
      fileType: ModelFileType.litertlm,
    )
        .fromNetwork(
      target.url,
      token: _blankToNull(token),
    )
        .withProgress((progress) {
      onProgress?.call(progress.toDouble() / 100);
    });
    if (cancelToken != null) {
      builder = builder.withCancelToken(cancelToken);
    }
    await builder.install();
  }

  @override
  Future<void> deleteModel({LlmModel? model}) async {
    final target = model ?? _activeModel;
    // If we're deleting what's currently loaded, drop the engine first.
    if (target == _activeModel) {
      await _persistedChat?.close();
      await _loadedModel?.close();
      _persistedChat = null;
      _persistedChatId = null;
      _persistedChatSamplingSig = null;
      _loadedModel = null;
      _loadedSupportsImage = false;
      _warmUpCompleter = null;
      _visionSwapCompleter = null;
    }
    await FlutterGemma.uninstallModel(target.filename);
  }

  @override
  bool get isWarm => _loadedModel != null;

  @override
  bool get hasVision => _loadedModel != null && _loadedSupportsImage;

  @override
  bool get isLoadingVision => _visionSwapCompleter != null;

  @override
  ReconfigureNeeds reconfigureNeedsFor({
    required String chatId,
    required InferenceSettings settings,
    required bool needsVision,
  }) {
    // Engine reload happens when maxTokens (clamped for vision) differs from
    // what's resident, OR no engine resident yet (cold path counts as reload
    // for UX purposes — user still sees the wait).
    final wantVision = needsVision || _loadedSupportsImage;
    final targetMaxTokens =
        wantVision && settings.maxTokens < _visionMinMaxTokens
            ? _visionMinMaxTokens
            : settings.maxTokens;
    final engineReload =
        _loadedModel == null || _loadedMaxTokens != targetMaxTokens;
    final visionLoad = needsVision && !_loadedSupportsImage;
    // Session rebuild only fires when an existing session needs to be torn
    // down: chatId switched OR sampling/RAG sig changed. First-ever session
    // creation (no _persistedChat yet) is NOT a "rebuild" from the user's
    // POV — they didn't change anything, the session simply hasn't been
    // born. Bundling that into sessionRebuild caused a spurious banner on
    // every first send. Eager-apply (called when the user actually changes
    // a slider) covers the "I tweaked params before first send" case
    // separately, so we don't need this clause to surface those changes.
    // If we'd be reloading the engine, the chat is rebuilt anyway — we
    // report sessionRebuild=false then so the banner shows the heavier
    // "engine reload" message, not both.
    final samplingSig = settings.samplingSignature;
    final sessionRebuild = !engineReload &&
        _persistedChat != null &&
        (_persistedChatId != chatId ||
            _persistedChatSamplingSig != samplingSig);
    return ReconfigureNeeds(
      engineReload: engineReload,
      visionLoad: visionLoad,
      sessionRebuild: sessionRebuild,
    );
  }

  @override
  AcceleratorChoice get accelerator => _accelerator;

  @override
  Future<void> setAccelerator(AcceleratorChoice choice) async {
    if (choice == _accelerator) return;
    debugPrint('[ash] accelerator change: $_accelerator → $choice — '
        'closing engine; next query will re-warm');
    _accelerator = choice;
    await closeEngine();
  }

  @override
  bool get speculativeDecoding => _speculativeDecoding;

  @override
  Future<void> setSpeculativeDecoding(bool enabled) async {
    if (enabled == _speculativeDecoding) return;
    debugPrint('[ash] speculative decoding: $_speculativeDecoding → '
        '$enabled — closing engine; next query will re-warm');
    _speculativeDecoding = enabled;
    await closeEngine();
  }

  PreferredBackend get _preferredBackend {
    // See [_gpuRebuildUnsafe] for the full story. tl;dr: GPU rebuild
    // crashes the process in the Metal delegate; force CPU once we've
    // ever closed the engine.
    if (_gpuRebuildUnsafe && _accelerator == AcceleratorChoice.gpu) {
      return PreferredBackend.cpu;
    }
    return switch (_accelerator) {
      AcceleratorChoice.gpu => PreferredBackend.gpu,
      AcceleratorChoice.cpu => PreferredBackend.cpu,
    };
  }

  @override
  Future<void> warmUp() async {
    // Default warm-up is text-only at the default maxTokens. The first
    // chat-scoped query will rebuild if its maxTokens differs.
    if (_loadedModel != null) return;
    if (_warmUpCompleter case Completer<void> c) return c.future;

    final completer = _warmUpCompleter = Completer<void>();
    try {
      debugPrint('[ash] warmUp: activating installed model');
      await _activateInstalledModel();
      debugPrint('[ash] warmUp: loading text-only engine '
          '(maxTok=$_defaultMaxTokens, backend=$_accelerator)');
      final sw = Stopwatch()..start();
      _loadedModel = await FlutterGemma.getActiveModel(
        maxTokens: _defaultMaxTokens,
        supportImage: false,
        supportAudio: false,
        preferredBackend: _preferredBackend,
        enableSpeculativeDecoding: _speculativeDecoding,
      );
      _loadedSupportsImage = false;
      _loadedMaxTokens = _defaultMaxTokens;
      debugPrint('[ash] warmUp: text engine ready in '
          '${sw.elapsedMilliseconds}ms');
      completer.complete();
    } catch (e, st) {
      debugPrint('[ash] warmUp failed: $e\n$st');
      _warmUpCompleter = null;
      completer.completeError(e, st);
      rethrow;
    }
  }

  /// Ensure the resident engine matches the request:
  ///   - vision-capable iff [needsVision] (or already-loaded vision, which we
  ///     keep sticky so a chat that used vision once doesn't pay the swap
  ///     again on subsequent text turns),
  ///   - KV-cache size = [requestedMaxTokens], clamped to ≥ [_visionMinMaxTokens]
  ///     if vision is required.
  ///
  /// Reuses the existing engine when both match. Otherwise closes and
  /// rebuilds. Surfaces a [_visionSwapCompleter] when the rebuild involves
  /// loading the vision encoder (slow Metal JIT, ~30s first time).
  Future<void> _ensureEngineFor({
    required int requestedMaxTokens,
    required bool needsVision,
  }) async {
    final wantVision = needsVision || _loadedSupportsImage; // sticky vision
    final targetMaxTokens =
        wantVision && requestedMaxTokens < _visionMinMaxTokens
            ? _visionMinMaxTokens
            : requestedMaxTokens;

    if (_loadedModel != null &&
        _loadedSupportsImage == wantVision &&
        _loadedMaxTokens == targetMaxTokens) {
      return; // already correct — fast path
    }

    // A rebuild involving vision loading deserves the UI banner.
    final isVisionLoad = wantVision && !_loadedSupportsImage;
    if (isVisionLoad) {
      final existing = _visionSwapCompleter;
      if (existing != null) return existing.future;
    }
    final completer =
        isVisionLoad ? (_visionSwapCompleter = Completer<void>()) : null;
    try {
      debugPrint('[ash] engine rebuild: '
          'wantVision=$wantVision maxTok=$targetMaxTokens '
          '(was vision=$_loadedSupportsImage maxTok=$_loadedMaxTokens)');
      await _persistedChat?.close();
      _persistedChat = null;
      _persistedChatId = null;
      _persistedChatSamplingSig = null;
      await _loadedModel?.close();
      _loadedModel = null;
      _loadedSupportsImage = false;
      _loadedMaxTokens = null;

      final sw = Stopwatch()..start();
      try {
        _loadedModel = await FlutterGemma.getActiveModel(
          maxTokens: targetMaxTokens,
          supportImage: wantVision,
          supportAudio: false,
          preferredBackend: _preferredBackend,
          enableSpeculativeDecoding: _speculativeDecoding,
        );
      } catch (e, st) {
        // Vision-engine creation on the GPU delegate intermittently fails
        // to prepare specific ops (we've seen `STABLEHLO_COMPOSITE` node
        // preparation failures in TestFlight-signed builds even though
        // the same model loads fine for text-only). Retry once on CPU
        // before giving up — the CPU delegate is slower but doesn't go
        // through Metal shader compilation, so it's a reliable fallback
        // for vision queries on devices where the GPU path is broken.
        //
        // Only re-attempt when (a) the failed attempt was vision-capable
        // (text-only on GPU has never failed for us, so retrying it on
        // CPU would just mask a different problem), and (b) we're not
        // already on CPU (no point retrying with the same delegate).
        if (!(wantVision && _preferredBackend == PreferredBackend.gpu)) {
          rethrow;
        }
        debugPrint('[ash] engine rebuild on GPU failed for vision — '
            'retrying with CPU backend: $e');
        _loadedModel = await FlutterGemma.getActiveModel(
          maxTokens: targetMaxTokens,
          supportImage: wantVision,
          supportAudio: false,
          preferredBackend: PreferredBackend.cpu,
          enableSpeculativeDecoding: _speculativeDecoding,
        );
        debugPrint('[ash] engine rebuild: CPU fallback succeeded '
            '(stack trace from GPU attempt was: $st)');
      }
      _loadedSupportsImage = wantVision;
      _loadedMaxTokens = targetMaxTokens;
      debugPrint('[ash] engine rebuild: ready in '
          '${sw.elapsedMilliseconds}ms');
      completer?.complete();
    } catch (e, st) {
      debugPrint('[ash] engine rebuild failed: $e\n$st');
      completer?.completeError(e, st);
      rethrow;
    } finally {
      if (isVisionLoad) _visionSwapCompleter = null;
    }
  }

  /// Make sure the resident engine + persisted chat session match [settings]
  /// for [chatId], creating/closing things as needed. Returns the ready-to-use
  /// [InferenceChat]. Used by both [query] (right before sending the user
  /// prompt) and [applySettings] (eager pre-warm on settings change).
  Future<InferenceChat> _prepareSession({
    required String chatId,
    required InferenceSettings settings,
    required List<HistoryTurn> priorHistory,
    required bool needsVision,
    required String systemPrompt,
  }) async {
    if (_loadedModel == null) {
      debugPrint('[ash] model not warm — loading '
          '${needsVision ? 'vision' : 'text'} engine now');
      await _activateInstalledModel();
    }
    await _ensureEngineFor(
      requestedMaxTokens: settings.maxTokens,
      needsVision: needsVision,
    );
    final model = _loadedModel!;

    // Chat persistence: reuse the same InferenceChat for successive turns
    // of the same conversation so the model sees history. The session is
    // invalidated when (a) chatId changes (user opened a different chat),
    // or (b) sampling params changed (the user adjusted settings — params
    // are baked at createChat time).
    final samplingSig = settings.samplingSignature;
    final mustRebuild = _persistedChat != null &&
        (_persistedChatId != chatId ||
            _persistedChatSamplingSig != samplingSig);
    if (mustRebuild) {
      debugPrint('[ash] chat session invalidated '
          '($_persistedChatId/$_persistedChatSamplingSig → '
          '$chatId/$samplingSig)');
      try {
        await _persistedChat?.close();
      } catch (_) {}
      _persistedChat = null;
      _persistedChatId = null;
      _persistedChatSamplingSig = null;
    }

    var chat = _persistedChat;
    final isFreshSession = chat == null;
    if (isFreshSession) {
      debugPrint('[ash] creating fresh chat session for $chatId '
          '(temp=${settings.temperature} topK=${settings.topK} '
          'topP=${settings.topP} lens=${_lensPackIds ?? 'all'})');
      chat = await model.createChat(
        temperature: settings.temperature,
        topK: settings.topK,
        topP: settings.topP,
        isThinking: settings.isThinking,
        modelType: ModelType.gemma4,
        supportImage: _loadedSupportsImage,
        supportAudio: false,
        systemInstruction: systemPrompt,
      );
      _persistedChat = chat;
      _persistedChatId = chatId;
      _persistedChatSamplingSig = samplingSig;

      // Replay prior turns into the new session so the model "remembers"
      // the conversation. Costs ~one prefill per turn; without this, the
      // model would see only the current user message after any engine
      // rebuild or chat-close (back-nav → closeEngine → re-entry).
      if (priorHistory.isNotEmpty) {
        debugPrint('[ash] replaying ${priorHistory.length} prior turn(s) '
            'into fresh session');
        for (final turn in priorHistory) {
          // Skip empty placeholders (assistant messages still streaming
          // at the time we snapshot the list).
          if (turn.text.isEmpty && turn.imageBytes == null) continue;
          final msg = (turn.imageBytes != null)
              ? Message.withImage(
                  text: turn.text,
                  imageBytes: turn.imageBytes!,
                  isUser: turn.isUser,
                )
              : Message.text(text: turn.text, isUser: turn.isUser);
          await chat.addQueryChunk(msg);
        }
      }
    } else {
      debugPrint('[ash] reusing chat session $chatId (multi-turn, '
          'ignoring ${priorHistory.length} replay turn(s))');
    }
    return chat;
  }

  @override
  Future<void> applySettings({
    required String chatId,
    required InferenceSettings settings,
    List<HistoryTurn> priorHistory = const [],
  }) async {
    // Pre-warm the engine + chat session so the next query streams without
    // a leading wait. We don't know if the next query will use vision, so
    // we honor whatever's already loaded (sticky vision) — this matches the
    // current UX: changing settings shouldn't drop the vision engine.
    final systemPrompt = _buildSystemPrompt(useRag: settings.useRag);
    await _prepareSession(
      chatId: chatId,
      settings: settings,
      priorHistory: priorHistory,
      needsVision: false, // sticky vision is respected inside _ensureEngineFor
      systemPrompt: systemPrompt,
    );
  }

  @override
  Future<void> closeEngine() async {
    // No-op if nothing loaded.
    if (_loadedModel == null &&
        _warmUpCompleter == null &&
        _visionSwapCompleter == null) {
      return;
    }
    // Flip the rebuild-unsafe flag BEFORE we actually close. The next
    // engine creation goes to CPU even if the close itself throws,
    // because the unstable Metal state is created by the prior open,
    // not by the close.
    _gpuRebuildUnsafe = true;
    debugPrint('[ash] closeEngine: releasing resident engine');
    try {
      await _persistedChat?.close();
    } catch (_) {}
    try {
      await _loadedModel?.close();
    } catch (_) {}
    _persistedChat = null;
    _persistedChatId = null;
    _persistedChatSamplingSig = null;
    _loadedModel = null;
    _loadedSupportsImage = false;
    _loadedMaxTokens = null;
    _warmUpCompleter = null;
    _visionSwapCompleter = null;
  }

  @override
  Future<void> resetChatSession() async {
    if (_persistedChat == null) return;
    debugPrint('[ash] resetChatSession: dropping chat KV — engine stays warm');
    try {
      await _persistedChat?.close();
    } catch (e) {
      debugPrint('[ash] resetChatSession close failed: $e');
    }
    _persistedChat = null;
    _persistedChatId = null;
    _persistedChatSamplingSig = null;
  }

  @override
  void setLens(Set<String>? lensPackIds) {
    if (lensPackIds == null || lensPackIds.isEmpty) {
      _lensPackIds = null;
    } else {
      _lensPackIds = Set.unmodifiable(lensPackIds);
    }
    debugPrint('[ash] retrieval lens = ${_lensPackIds ?? 'all installed'}');
  }

  @override
  Stream<InferenceChunk> query({
    required String chatId,
    required String prompt,
    Uint8List? imageBytes,
    List<HistoryTurn> priorHistory = const [],
    InferenceSettings settings = InferenceSettings.defaults,
    bool liveMode = false,
  }) async* {
    _stopRequested = false;

    final usesImage = imageBytes != null;
    debugPrint('[ash] query start: chatId=$chatId image=$usesImage '
        'promptLen=${prompt.length} rag=${settings.useRag} '
        'temp=${settings.temperature} topK=${settings.topK} '
        'topP=${settings.topP} maxTok=${settings.maxTokens} '
        'backend=$_accelerator');

    try {
      String fullPrompt;
      List<MessageSource> sources = const [];
      if (usesImage || !settings.useRag) {
        // Image queries skip RAG (context budget) and the user can also
        // disable RAG entirely via settings.
        fullPrompt = prompt;
      } else {
        await _ensureDb();
        // Pass 1: embed the raw user prompt and retrieve.
        List<double> queryEmbedding;
        try {
          queryEmbedding = await _embedQuery(prompt);
        } catch (e) {
          debugPrint('[ash] Embedding failed: $e');
          queryEmbedding = [];
        }
        var hybrid = queryEmbedding.isNotEmpty
            ? await _retrieveHybrid(queryEmbedding, prompt, _retrieveTopK)
            : (chunks: <RetrievedInfo>[], bestVectorScore: null);
        var retrieved = hybrid.chunks;
        final pass1VectorScore = hybrid.bestVectorScore;
        debugPrint('[ash] pass-1 retrieved ${retrieved.length} chunks '
            '(top vector score: '
            '${pass1VectorScore?.toStringAsFixed(3) ?? 'n/a'})');

        // Pass 2 (adaptive): if pass 1 was weak (no chunks, or top
        // chunk's cosine distance is > _rewriteTriggerScore meaning it's a
        // sentence-structure match more than a topic match), ask Gemma
        // to rewrite the query into a search-ready question, then
        // retrieve again with that. Keep whichever pass got the better
        // vector score.
        //
        // Skip the rewriter for live (voice) mode. Spoken queries are
        // already verbose enough to retrieve well on the first pass,
        // and the rewriter's chat-session juggling (close user chat →
        // create rewriter chat → close it → rebuild user chat) causes
        // FFI crashes when the user interrupts during the transitional
        // window. Net: live mode trades a marginal retrieval quality
        // gain for a much more stable audio loop.
        final weakFirstPass = retrieved.isEmpty ||
            (pass1VectorScore ?? double.infinity) > _rewriteTriggerScore;
        if (weakFirstPass && !liveMode) {
          try {
            final rewritten = await _rewriteQueryForRetrieval(prompt);
            if (rewritten.trim().isNotEmpty &&
                rewritten.trim().toLowerCase() != prompt.trim().toLowerCase()) {
              final rewEmb = await _embedQuery(rewritten);
              if (rewEmb.isNotEmpty) {
                final rewHybrid =
                    await _retrieveHybrid(rewEmb, rewritten, _retrieveTopK);
                final rewBestVector =
                    rewHybrid.bestVectorScore ?? double.infinity;
                final rawBestVector = pass1VectorScore ?? double.infinity;
                if (rewBestVector < rawBestVector) {
                  debugPrint(
                      '[ash] pass-2 improved retrieval: top vector score '
                      '$rewBestVector (was $rawBestVector)');
                  retrieved = rewHybrid.chunks;
                }
              }
            }
          } catch (e) {
            debugPrint('[ash] adaptive rewrite path failed: $e');
          }
        }
        debugPrint('[ash] final retrieved ${retrieved.length} chunks');
        final assembled = _assembleContext(retrieved, prompt);
        sources = assembled.sources;
        // Soft framing: context first, then the user message verbatim. We
        // intentionally do NOT add a "User:" prefix — the Gemma 4 chat
        // template already wraps this whole string as a `<start_of_turn>user`
        // block, so an inner "User:" would be a duplicate role marker and
        // tends to confuse the model (truncated / off-topic replies). The
        // system instruction tells the model when to use the material.
        fullPrompt =
            assembled.text.isEmpty ? prompt : '${assembled.text}\n\n$prompt';
      }

      // Build system prompt based on (a) whether RAG is enabled, (b) the
      // active pack, (c) whether the caller is in live voice mode, and
      // (d) whether this is a vision query. Vision queries get a short
      // visual-task prompt with NO citation directive — the RAG-aware
      // prompt's "cite as [N]" instruction adds noise to image
      // understanding and was making Gemma 4 less attentive to the
      // actual photo.
      final systemPrompt = _buildSystemPrompt(
        useRag: settings.useRag,
        liveMode: liveMode,
        usesImage: usesImage,
      );

      final chat = await _prepareSession(
        chatId: chatId,
        settings: settings,
        priorHistory: priorHistory,
        needsVision: usesImage,
        systemPrompt: systemPrompt,
      );

      // Publish the active chat BEFORE addQueryChunk. Prefill runs inside
      // generateChatResponseAsync's first iteration, but the model is
      // already "engaged" the moment addQueryChunk has fed the prompt —
      // stop() needs a handle to call stopGeneration() if the user
      // interrupts during the long-prefill window.
      _activeChat = chat;
      _stopGenerationCalled = false;
      _generating = true;

      final message = usesImage
          ? Message.withImage(
              text: fullPrompt,
              imageBytes: imageBytes,
              isUser: true,
            )
          : Message.text(text: fullPrompt, isUser: true);
      debugPrint('[ash] adding query chunk (image=$usesImage)');
      await chat.addQueryChunk(message);

      // Preflight: hand the UI the list of sources we just baked into the
      // prompt, BEFORE any tokens stream. The caller stores them on the
      // streaming assistant message so the citation chips appear up front
      // (rather than popping in once the reply ends). Empty list ⇒ no
      // chunks cleared the threshold — emit nothing.
      if (sources.isNotEmpty) {
        yield InferenceChunk(text: '', isDone: false, sources: sources);
      }

      debugPrint('[ash] streaming response...');

      var accumulated = '';
      // On iOS .litertlm, the SDK does NOT strip turn markers from the
      // streamed tokens — and `<end_of_turn>` typically arrives FRAGMENTED
      // across several subword tokens (`<`, `end`, `_of`, `_turn`, `>`).
      // So we can't recognize EOS at the per-token level. Instead we keep a
      // rolling buffer of recent text, look for the marker as a substring,
      // and stop generation the moment it appears. Tokens that haven't been
      // confirmed safe (i.e. could still be the start of an EOS marker) are
      // held back until either more tokens disambiguate or generation ends.
      //
      // Without this guard the engine decodes past EOS and drifts into
      // pretraining noise — random multilingual text, code, repeated
      // punctuation. This was the "spits out different languages" bug.
      const stopMarkers = <String>[
        '<end_of_turn>',
        '<eos>',
        '<start_of_turn>', // model accidentally starting a new turn — also stop
      ];
      // Longest stop marker, used to size the "held back" tail buffer.
      final maxStopLen =
          stopMarkers.fold<int>(0, (m, s) => s.length > m ? s.length : m);
      var pending = '';
      var stopHit = false;
      String emitSafe(String chunk) {
        pending += chunk;
        // Earliest occurrence of any stop marker in the pending buffer.
        var earliest = -1;
        for (final marker in stopMarkers) {
          final idx = pending.indexOf(marker);
          if (idx >= 0 && (earliest < 0 || idx < earliest)) earliest = idx;
        }
        if (earliest >= 0) {
          final safe = pending.substring(0, earliest);
          pending = '';
          stopHit = true;
          return safe;
        }
        // No full marker. Hold back the tail that could still be the
        // beginning of one ("<", "<end", "<end_of_t…"), emit the rest.
        var keep = 0;
        for (var n = maxStopLen - 1; n >= 1; n--) {
          if (n > pending.length) continue;
          final tail = pending.substring(pending.length - n);
          if (stopMarkers.any((m) => m.startsWith(tail))) {
            keep = n;
            break;
          }
        }
        final safe = pending.substring(0, pending.length - keep);
        pending = pending.substring(pending.length - keep);
        return safe;
      }

      await for (final response in chat.generateChatResponseAsync()) {
        if (_stopRequested) {
          if (!_stopGenerationCalled) {
            _stopGenerationCalled = true;
            await chat.stopGeneration();
          }
          break;
        }

        if (response is TextResponse) {
          final token = response.token;
          // Filter out solo special tokens (<unk>, <bos>, <pad>, <unused*>).
          // EOS detection happens via the buffer below — not here.
          if (!_isUserVisibleToken(token)) continue;

          final safe = emitSafe(token);
          if (safe.isNotEmpty) {
            accumulated += safe;
            yield InferenceChunk(text: safe, isDone: false);
          }

          if (stopHit) {
            debugPrint('[ash] turn-boundary marker detected — '
                'stopping generation. Accumulated len=${accumulated.length}');
            if (!_stopGenerationCalled) {
              _stopGenerationCalled = true;
              await chat.stopGeneration();
            }
            break;
          }
        }
      }
      // Flush anything in the pending tail that turned out NOT to be part of
      // a stop marker (e.g. a literal "<" at the end of the reply).
      if (!stopHit && pending.isNotEmpty) {
        accumulated += pending;
        yield InferenceChunk(text: pending, isDone: false);
      }

      // Debug: log the full assistant response so we can audit it from the
      // flutter run terminal. Truncate to keep the log readable.
      final logSnippet = accumulated.length > 400
          ? '${accumulated.substring(0, 400)}…(${accumulated.length} chars)'
          : accumulated;
      debugPrint('[ash] assistant reply: $logSnippet');

      yield const InferenceChunk(text: '', isDone: true);
    } catch (error, stackTrace) {
      debugPrint('[ash] Query error: $error\n$stackTrace');
      // Previously this swallowed the error and yielded an empty "done"
      // chunk — which caused the UI to render a blank assistant bubble
      // with no indication of what went wrong. Rethrow so the consumer's
      // try/catch in _runInference fires and shows the user a
      // "Something went wrong" message instead of nothing.
      rethrow;
    } finally {
      // CRITICAL SAFETY NET. If we exited the await-for via consumer
      // cancellation (the StreamSubscription was cancelled by the UI)
      // OR via an unrelated exception, the native LiteRT-LM iterator
      // is STILL running — Dart-side cancellation only abandons the
      // pull side. Without calling stopGeneration() here, native keeps
      // producing tokens, eventually tries to push to the now-closed
      // StreamController, and segfaults the process ~1s later. The
      // idempotency guard means this is a no-op if the in-loop break
      // already fired the call.
      final chat = _activeChat;
      if (chat != null && !_stopGenerationCalled) {
        _stopGenerationCalled = true;
        try {
          await chat.stopGeneration();
        } catch (e) {
          debugPrint('[ash] stopGeneration in finally failed: $e');
        }
      }
      _activeChat = null;
      _generating = false;
      _stopRequested = false;
    }
    // Note: do NOT close _persistedChat here — it must survive across turns.
    // closeEngine() releases it when the user leaves the chat surface.
  }

  @override
  Future<void> stop() async {
    // Two-layer stop. The flag handles the between-tokens case (cheap,
    // gets caught at the top of the next iteration). The direct
    // stopGeneration() call handles the PREFILL case (long native
    // forward-pass over the prompt where the await-for hasn't yielded
    // yet so the flag check never executes) AND the CONSUMER-CANCELLED
    // case (where the loop is aborted by Dart-side stream cancellation
    // before the check runs).
    //
    // The `_stopGenerationCalled` idempotency guard ensures the in-loop
    // path, the finally-block safety net, and this direct path together
    // produce AT MOST ONE stopGeneration() call per generation. Double
    // calls into native were the previous theory's segfault source.
    //
    // If no chat is active (generation completed or never started),
    // skip the FFI call entirely — calling stopGeneration() on a
    // disposed/idle chat is what segfaults iOS LiteRT-LM.
    _stopRequested = true;
    final chat = _activeChat;
    if (chat == null || _stopGenerationCalled || !_generating) return;
    _stopGenerationCalled = true;
    try {
      // Bound the wait so a hung native call can't freeze the UI.
      // stopGeneration is meant to return promptly; 1.5s is generous.
      await chat.stopGeneration().timeout(
        const Duration(milliseconds: 1500),
        onTimeout: () {
          debugPrint('[ash] stopGeneration timed out');
        },
      );
    } catch (e) {
      debugPrint('[ash] direct stopGeneration failed: $e');
    }
  }

  @override
  bool get isGenerating => _generating;

  // --- Pack management ---

  @override
  Future<void> importPack(String packId) async {
    await _ensureDb();

    final assetPath = '$_packsDir/$packId.json';
    String jsonStr;
    try {
      jsonStr = await rootBundle.loadString(assetPath);
    } on FlutterError {
      debugPrint('[ash] Pack asset not found: $assetPath');
      return;
    }

    _importChunksFromJson(packId, jsonStr);
  }

  @override
  Future<void> importPackFromUrl(
    String packId,
    String url, {
    void Function(double progress)? onProgress,
  }) async {
    await _ensureDb();
    debugPrint('[ash] Downloading pack $packId from $url');
    final client = HttpClient();
    try {
      final req = await client.getUrl(Uri.parse(url));
      final resp = await req.close();
      if (resp.statusCode != 200) {
        throw Exception(
            'HTTP ${resp.statusCode} fetching pack $packId from $url');
      }
      final total = resp.contentLength; // -1 when chunked
      final bytes = <int>[];
      var got = 0;
      await for (final chunk in resp) {
        bytes.addAll(chunk);
        got += chunk.length;
        if (onProgress != null) {
          // Cap visible progress at 0.95 — the last 5% is reserved for the
          // ObjectBox put which lands on the next event-loop tick.
          final p = total > 0
              ? (got / total) * 0.95
              : 0.5; // unknown length: show indeterminate-ish midpoint
          onProgress(p.clamp(0.0, 0.95));
        }
      }
      // TODO(remote-packs): when remote distribution actually lands, add
      // SHA-256 verification here (requires the `crypto` package). For now
      // the bundle path is the trust boundary.
      _importChunksFromJson(packId, utf8.decode(bytes));
      onProgress?.call(1.0);
    } finally {
      client.close();
    }
  }

  /// Parse a pack JSON string and insert its chunks into ObjectBox. Shared
  /// by [importPack] (rootBundle) and [importPackFromUrl] (HTTP).
  void _importChunksFromJson(String packId, String jsonStr) {
    // Bust any cached centroid for this pack so the next rank call
    // recomputes against the freshly-installed chunks.
    _packCentroids.remove(packId);
    _bm25.invalidate();
    try {
      final data = jsonDecode(jsonStr) as Map<String, dynamic>;
      final packName = data['packName'] as String?;
      if (packName != null && packName.isNotEmpty) {
        _packIdToName[packId] = packName;
      }
      final List<dynamic> items = data['chunks'] as List<dynamic>;
      debugPrint('[ash] Importing pack $packId: ${items.length} chunks');
      final entities = <ChunkEntity>[];
      for (var i = 0; i < items.length; i++) {
        final m = items[i] as Map<String, dynamic>;
        final embRaw = m['embedding'] as List?;
        final embedding = embRaw?.map((e) => (e as num).toDouble()).toList();
        entities.add(
          ChunkEntity(
            id: 0,
            chunkId: m['chunkId'] as String? ?? '',
            text: m['text'] as String? ?? '',
            source: m['source'] as String? ?? '',
            sectionPath: m['sectionPath'] as String?,
            sourceType: m['sourceType'] as String? ?? 'core',
            packId: packId,
            embeddingModelVersion: m['embeddingModelVersion'] as String?,
            embedding: embedding,
          ),
        );
      }
      _box!.putMany(entities);
      debugPrint('[ash] Imported ${entities.length} chunks for pack $packId');
    } catch (e, stackTrace) {
      debugPrint('[ash] Failed to import pack $packId: $e\n$stackTrace');
    }
  }

  /// Cached LLM query rewrites — `raw user prompt → search-ready question`.
  /// MiniLM struggles with terse statement-form queries (the user types
  /// "i got shot"; the corpus's nearest neighbor is a fire chunk because
  /// both share "What should I do if X" structure). Adaptive rewrite
  /// reshapes the user's message into the form the index was built
  /// around. Bounded — see [_maxRewriteCacheSize].
  final Map<String, String> _rewriteCache = {};
  static const _maxRewriteCacheSize = 100;

  /// Score above which we'll try a query rewrite. ObjectBox returns
  /// cosine distance — lower is better. ≤0.5 means we already have a
  /// confident match; anything looser fires the rewrite path.
  static const _rewriteTriggerScore = 0.5;

  /// System prompt for the rewriter chat. Asks Gemma to produce a
  /// concise question that matches how the index's stored questions
  /// are phrased ("How do I treat X?", "What should I do if Y?"). The
  /// rewrite is used for RETRIEVAL only; the user's original prompt is
  /// what gets sent to the model for the actual answer, so phrasing
  /// fidelity to the user's voice is preserved.
  ///
  /// The "NO_REWRITE" sentinel is the safety valve: when the user's
  /// message is a greeting / smalltalk / has no extractable topic, we
  /// don't want Gemma to hallucinate a medical query out of thin air
  /// (the "type 'hi' and get a wound citation" bug).
  static const _rewriterSystemPrompt =
      'You rewrite a user\'s message into a single concise question they '
      'might type into a survival-knowledge search bar. \n'
      'Rules:\n'
      '1. If the message is a greeting, thanks, smalltalk, or has no '
      'clear topic (e.g. "hi", "hello", "help", "thanks", "what", "ok"), '
      'output exactly: NO_REWRITE\n'
      '2. If the message is already a clear specific question, repeat it '
      'verbatim.\n'
      '3. Otherwise, rewrite it into one concise question. Only use '
      'specific medical/event terms (gunshot, tourniquet, hypothermia, '
      'etc.) when the user\'s message clearly implies them — never '
      'invent a topic the user didn\'t mention.\n'
      'Output exactly one line: either NO_REWRITE or the rewritten '
      'question. No preamble, no explanation.';

  /// Cache of per-pack representative embeddings, keyed by packId. For
  /// installed packs we use the L2-normalized mean of every chunk's
  /// embedding (a "centroid") — that's a much better semantic signature
  /// than embedding the synthetic summary text, which was basically
  /// `"{Name} emergency preparedness and response guide."` for every pack
  /// and made the ranker return noise. For uninstalled packs (no chunks
  /// in DB), we fall back to embedding the [texts] entry the caller
  /// provided. Cache is invalidated by [closeEngine] / pack uninstall.
  final Map<String, List<double>> _packCentroids = {};

  @override
  Future<List<PackRanking>> rankByQuery({
    required String query,
    required Map<String, String> texts,
  }) async {
    if (query.trim().isEmpty || texts.isEmpty) return const [];
    await _ensureEmbedding();
    await _ensureDb();
    final qEmb = await _embedQuery(query);
    if (qEmb.isEmpty) return const [];

    final results = <PackRanking>[];
    for (final entry in texts.entries) {
      final packId = entry.key;
      var emb = _packCentroids[packId];
      if (emb == null) {
        emb = await _computePackCentroid(packId);
        if (emb == null) {
          // Pack has no chunks in DB (uninstalled) — fall back to embedding
          // the human-readable text the caller passed in. Lower-quality
          // ranking signal but at least it's something.
          try {
            emb = await _embedQuery(entry.value);
          } catch (e) {
            debugPrint('[ash] pack-rank fallback embed failed for $packId: $e');
            continue;
          }
        }
        _packCentroids[packId] = emb;
      }
      // Cosine similarity for two L2-normalized vectors = dot product.
      var dot = 0.0;
      for (var i = 0; i < _embeddingDimensions; i++) {
        dot += qEmb[i] * emb[i];
      }
      results.add(PackRanking(packId: packId, score: dot));
    }
    results.sort((a, b) => b.score.compareTo(a.score));
    return results;
  }

  /// Compute the L2-normalized mean of every chunk embedding for [packId].
  /// Returns null when the pack has no chunks in the vector store yet
  /// (e.g. user hasn't installed it). Microseconds per pack at hackathon
  /// scale (~10–30 chunks each).
  Future<List<double>?> _computePackCentroid(String packId) async {
    if (_box == null) return null;
    final q = _box!.query(ChunkEntity_.packId.equals(packId)).build();
    final chunks = q.find();
    q.close();
    if (chunks.isEmpty) return null;
    final centroid = List<double>.filled(_embeddingDimensions, 0.0);
    var count = 0;
    for (final c in chunks) {
      final emb = c.embedding;
      if (emb == null || emb.length != _embeddingDimensions) continue;
      for (var i = 0; i < _embeddingDimensions; i++) {
        centroid[i] += emb[i];
      }
      count++;
    }
    if (count == 0) return null;
    // Mean. The mean of unit vectors isn't unit-length, so re-normalize.
    var norm = 0.0;
    for (var i = 0; i < _embeddingDimensions; i++) {
      centroid[i] /= count;
      norm += centroid[i] * centroid[i];
    }
    norm = sqrt(norm);
    if (norm > 0) {
      for (var i = 0; i < _embeddingDimensions; i++) {
        centroid[i] /= norm;
      }
    }
    return centroid;
  }

  @override
  Future<List<LibraryChunk>> readPack(String packId) async {
    await _ensureDb();
    if (_box == null) return const [];
    final q = _box!.query(ChunkEntity_.packId.equals(packId)).build();
    final results = q.find();
    q.close();
    // ObjectBox preserves insertion order via primary key — matches the
    // original markdown source order in the JSON pack. Sanitize as we go
    // so the Library shows the same cleaned text the model sees.
    return [
      for (final e in results)
        LibraryChunk(
          chunkId: e.chunkId,
          text: sanitizeChunkText(e.text),
          sectionPath: e.sectionPath,
        ),
    ];
  }

  @override
  Future<void> uninstallPack(String packId) async {
    await _ensureDb();

    final query = _box!.query(ChunkEntity_.packId.equals(packId)).build();
    final count = query.remove();
    query.close();
    // Centroid for this pack is no longer valid — drop it so the next
    // rank call either recomputes (if the pack is reinstalled) or falls
    // back to the text-embedding path.
    _packCentroids.remove(packId);
    _bm25.invalidate();
    debugPrint('[ash] Uninstalled pack $packId: removed $count chunks');
  }

  @override
  Future<Set<String>> installedPackIds() async {
    await _ensureDb();
    if (_box == null || _box!.count() == 0) return <String>{};
    // ObjectBox doesn't expose DISTINCT directly; iterate the index. With
    // ~1k chunks total this is fine — runs in microseconds.
    final all = _box!.getAll();
    final ids = <String>{};
    for (final c in all) {
      if (c.packId.isNotEmpty) ids.add(c.packId);
    }
    return ids;
  }

  @override
  Future<void> importAllPacks() async {
    await _ensureDb();

    // Read registry to get all pack IDs
    String registryStr;
    try {
      registryStr = await rootBundle.loadString(_registryAsset);
    } on FlutterError {
      debugPrint('[ash] packs_registry.json not found');
      return;
    }

    try {
      final List<dynamic> registry = jsonDecode(registryStr);
      for (final entry in registry) {
        final m = entry as Map<String, dynamic>;
        final packId = m['packId'] as String;
        // Populate the packId → name cache eagerly from the registry so
        // retrieval can label source blocks even when the pack was imported
        // in a previous session (no in-memory state from importPack).
        final packName = m['packName'] as String?;
        if (packName != null && packName.isNotEmpty) {
          _packIdToName[packId] = packName;
        }
        // Skip if already imported
        final existing =
            _box!.query(ChunkEntity_.packId.equals(packId)).build().count();
        if (existing > 0) {
          debugPrint('[ash] Pack $packId already imported, skipping');
          continue;
        }
        await importPack(packId);
      }
    } catch (e, stackTrace) {
      debugPrint('[ash] Failed to import all packs: $e\n$stackTrace');
    }
  }

  // --- ObjectBox ---

  Future<void> _ensureDb() async {
    if (_store != null) return;

    final docsDir = await getApplicationDocumentsDirectory();
    final dbDir = '${docsDir.path}/$_dbSubdirectory';

    final dir = Directory(dbDir);
    if (!await dir.exists()) {
      await dir.create(recursive: true);
    }

    try {
      _store = await openStore(directory: dbDir);
    } catch (e) {
      // Schema mismatch from a previous version — wipe and recreate.
      debugPrint('[ash] ObjectBox schema mismatch, resetting DB: $e');
      await dir.delete(recursive: true);
      await dir.create(recursive: true);
      _store = await openStore(directory: dbDir);
    }
    _box = _store!.box<ChunkEntity>();
  }

  // --- Embedding ---

  Future<void> _ensureEmbedding() async {
    if (_embeddingSession != null) return;

    _ort = OnnxRuntime();
    _embeddingSession = await _ort!.createSessionFromAsset(
      _embeddingModelAsset,
    );

    if (_tokenizer == null) {
      final vocabData = await rootBundle.loadString(_vocabAsset);
      final vocabLines = vocabData.split('\n');
      _tokenizer = BertTokenizer(vocabLines);
    }
  }

  /// Greetings / smalltalk / single-syllable acks. Don't even bother
  /// invoking the rewriter for these — the LLM happily fabricates a
  /// medical query for "hi" because the rewriter prompt nudges it
  /// toward survival topics. Fast path returns the raw prompt without
  /// any Gemma round-trip.
  static const _nonSubstantiveQueries = <String>{
    'hi',
    'hello',
    'hey',
    'howdy',
    'yo',
    'sup',
    'thanks',
    'thank you',
    'thx',
    'ty',
    'cheers',
    'bye',
    'goodbye',
    'see ya',
    'later',
    'ok',
    'okay',
    'k',
    'sure',
    'alright',
    'yes',
    'yeah',
    'yep',
    'no',
    'nope',
    'maybe',
    'cool',
    'nice',
    'great',
    'awesome',
    'wait',
    'hold on',
    'one sec',
    'help',
    'what',
    'huh',
    '?',
    '??',
  };

  bool _isNonSubstantive(String prompt) {
    final t = prompt.trim().toLowerCase().replaceAll(RegExp(r'[!.?,]+$'), '');
    if (t.isEmpty) return true;
    if (_nonSubstantiveQueries.contains(t)) return true;
    // 1- or 2-word inputs starting with a greeting are also smalltalk
    // ("hi there", "hello ash", "thanks a lot").
    if (t.startsWith('hi ') ||
        t.startsWith('hello ') ||
        t.startsWith('hey ') ||
        t.startsWith('thanks ') ||
        t.startsWith('how are you') ||
        t.startsWith('what\'s up')) {
      return true;
    }
    return false;
  }

  /// Run a short Gemma forward pass to rewrite a terse user query into
  /// a search-ready question. Used adaptively — only fired when the
  /// raw-query retrieval came back weak (no chunks below
  /// [_rewriteTriggerScore] cosine distance). Returns the original
  /// prompt on any failure so the caller's retrieval path stays
  /// graceful.
  ///
  /// The rewriter uses an EPHEMERAL chat session that's torn down
  /// immediately so it doesn't interfere with the user's persisted
  /// chat (separate KV cache; same engine).
  Future<String> _rewriteQueryForRetrieval(String prompt) async {
    final cleaned = prompt.trim();
    if (cleaned.isEmpty) return prompt;
    // Heuristic gate: greetings / smalltalk skip the round-trip
    // entirely. Caching the raw prompt as its own rewrite means future
    // identical inputs short-circuit without hitting the LLM either.
    if (_isNonSubstantive(cleaned)) {
      _rewriteCache[cleaned] = cleaned;
      return cleaned;
    }
    // Cache hit?
    final cached = _rewriteCache[cleaned];
    if (cached != null) return cached;
    // No engine yet — skip the rewrite, the caller will fall back to
    // raw retrieval. The engine warms on the first user turn; rewrites
    // start helping from turn 2 onward.
    if (_loadedModel == null) return prompt;

    // iOS LiteRT-LM serializes chat sessions through a single underlying
    // engine state. Creating a fresh `rewriterChat` while the user's
    // `_persistedChat` is still around leaves the persisted chat with a
    // stale session handle — its next generate call hangs silently
    // (observed: "i got shot" producing an empty assistant bubble).
    //
    // Pre-emptively close the user's chat so the rewriter owns the
    // session cleanly. `_prepareSession` later in query() sees
    // `_persistedChat == null` and rebuilds with history replay — costs
    // ~50-100 ms per prior turn, but the alternative is the pipeline
    // hanging.
    if (_persistedChat != null) {
      debugPrint('[ash] rewrite: closing user chat to free up session');
      try {
        await _persistedChat?.close();
      } catch (_) {}
      _persistedChat = null;
      _persistedChatId = null;
      _persistedChatSamplingSig = null;
    }

    InferenceChat? rewriterChat;
    try {
      rewriterChat = await _loadedModel!.createChat(
        // Mostly greedy — we want consistent rewrites for the same
        // input across runs (so the cache is meaningful).
        temperature: 0.2,
        topK: 40,
        topP: 0.95,
        isThinking: false,
        modelType: ModelType.gemma4,
        supportImage: false,
        supportAudio: false,
        systemInstruction: _rewriterSystemPrompt,
      );
      await rewriterChat.addQueryChunk(
        Message.text(text: cleaned, isUser: true),
      );
      final sb = StringBuffer();
      // Hard wall-clock timeout. If the rewriter hangs (rare but
      // observed once), TimeoutException propagates up to the catch
      // block and the caller falls back to the raw query path. Without
      // this guard a stuck rewriter blocks every subsequent user turn.
      await rewriterChat.generateChatResponseAsync().forEach((r) {
        if (r is TextResponse) {
          final token = r.token;
          // Drop turn-marker fragments the SDK occasionally leaks.
          if (_specialTokenRegex.hasMatch(token.trim())) return;
          sb.write(token);
          final acc = sb.toString();
          // Stop early on first '?' or newline — that's the rewrite.
          // Safety cap so a rambling model can't pin the rewriter.
          if (acc.contains('?') || acc.contains('\n') || sb.length > 200) {
            rewriterChat?.stopGeneration();
          }
        }
      }).timeout(const Duration(seconds: 5));
      var rewrite = sb.toString();
      // First line only.
      final nl = rewrite.indexOf('\n');
      if (nl > 0) rewrite = rewrite.substring(0, nl);
      // Strip any stray markup that survived the special-token regex.
      rewrite = rewrite
          .replaceAll(RegExp(r'<[^>]+>'), '')
          .replaceAll(RegExp(r'\*+'), '')
          .trim();
      // Sentinel: Gemma decided the input has no extractable topic
      // (greeting, smalltalk, ambiguous "help"). Fall back to the raw
      // prompt — which will simply produce no retrieval hits, which is
      // the correct behavior for "hi".
      if (rewrite.toUpperCase().contains('NO_REWRITE')) {
        _rewriteCache[cleaned] = cleaned;
        return cleaned;
      }
      // Reject empty / extreme outputs — fall back to the raw prompt.
      if (rewrite.isEmpty || rewrite.length > 200) return prompt;

      // Bounded cache: when full, evict the oldest entry. Map preserves
      // insertion order in Dart, so first key = oldest.
      if (_rewriteCache.length >= _maxRewriteCacheSize) {
        _rewriteCache.remove(_rewriteCache.keys.first);
      }
      _rewriteCache[cleaned] = rewrite;
      debugPrint('[ash] rewrite: "$cleaned" → "$rewrite"');
      return rewrite;
    } catch (e) {
      debugPrint('[ash] rewrite failed: $e — using raw query');
      return prompt;
    } finally {
      try {
        await rewriterChat?.close();
      } catch (_) {}
    }
  }

  Future<List<double>> _embedQuery(String query) async {
    await _ensureEmbedding();

    final result = _tokenizer!.tokenize(query, maxLength: _maxSeqLength);
    final inputIds = result.inputIds;
    final attentionMask = result.attentionMask;

    final inputIdsTensor = await OrtValue.fromList(
      Int64List.fromList(inputIds),
      [1, _maxSeqLength],
    );
    final maskTensor = await OrtValue.fromList(
      Int64List.fromList(attentionMask),
      [1, _maxSeqLength],
    );
    final tokenTypeTensor = await OrtValue.fromList(
      Int64List.fromList(List<int>.filled(_maxSeqLength, 0)),
      [1, _maxSeqLength],
    );

    OrtValue? outputTensor;
    try {
      final outputs = await _embeddingSession!.run({
        'input_ids': inputIdsTensor,
        'attention_mask': maskTensor,
        'token_type_ids': tokenTypeTensor,
      });

      outputTensor = outputs['last_hidden_state'];

      final flatFloats = (await outputTensor!.asFlattenedList()).cast<double>();

      // Mean pooling over token dimension.
      // Output shape: [1, seq_len, 384]
      final embedding = List<double>.filled(_embeddingDimensions, 0.0);
      var validTokens = 0;
      for (var t = 0; t < _maxSeqLength; t++) {
        if (attentionMask[t] == 0) continue;
        validTokens++;
        for (var d = 0; d < _embeddingDimensions; d++) {
          embedding[d] += flatFloats[t * _embeddingDimensions + d];
        }
      }
      if (validTokens > 0) {
        for (var d = 0; d < _embeddingDimensions; d++) {
          embedding[d] /= validTokens;
        }
      }

      // L2 normalize
      final norm = sqrt(embedding.fold(0.0, (sum, v) => sum + v * v));
      if (norm > 0) {
        for (var d = 0; d < _embeddingDimensions; d++) {
          embedding[d] /= norm;
        }
      }

      // Dispose output tensors
      for (final t in outputs.values) {
        await t.dispose();
      }

      return embedding;
    } finally {
      await inputIdsTensor.dispose();
      await maskTensor.dispose();
      await tokenTypeTensor.dispose();
    }
  }

  // --- Retrieval ---

  Future<List<RetrievedInfo>> _retrieveVector(
    List<double> queryEmbedding,
    int topK,
  ) async {
    await _ensureDb();
    if (_box == null || _box!.count() == 0) return [];

    // Build query — scope to the active lens if set. Three shapes:
    //   • null/empty   → search every installed pack
    //   • exactly one  → equals (fast, single-property index lookup)
    //   • many         → oneOf (ObjectBox compiles to a multi-equals OR)
    final lens = _lensPackIds;
    final QueryBuilder<ChunkEntity> qb;
    if (lens == null || lens.isEmpty) {
      qb = _box!.query(
        ChunkEntity_.embedding.nearestNeighborsF32(queryEmbedding, topK),
      );
    } else if (lens.length == 1) {
      qb = _box!.query(
        ChunkEntity_.embedding
            .nearestNeighborsF32(queryEmbedding, topK)
            .and(ChunkEntity_.packId.equals(lens.first)),
      );
    } else {
      qb = _box!.query(
        ChunkEntity_.embedding
            .nearestNeighborsF32(queryEmbedding, topK)
            .and(ChunkEntity_.packId.oneOf(lens.toList())),
      );
    }
    final query = qb.build();

    final results = query.findWithScores();
    query.close();

    // Threshold-filter so off-topic queries don't get spurious context.
    // ObjectBox returns cosine distance — keep results below the ceiling.
    // Sort ascending by score so the strongest matches end up at [0].
    final filtered = [
      for (final r in results)
        if (r.score <= _maxScoreThreshold) r,
    ]..sort((a, b) => a.score.compareTo(b.score));

    if (results.length != filtered.length) {
      debugPrint('[ash] retrieve: ${results.length} candidates → '
          '${filtered.length} after threshold (≤$_maxScoreThreshold). '
          'Worst kept: ${filtered.isEmpty ? '—' : filtered.last.score.toStringAsFixed(3)}, '
          'best dropped: ${results.length > filtered.length ? results.where((r) => r.score > _maxScoreThreshold).map((r) => r.score).reduce(min).toStringAsFixed(3) : '—'}');
    }

    return [
      for (final r in filtered)
        RetrievedInfo(
          chunkId: r.object.chunkId,
          text: r.object.text,
          source: r.object.source,
          sourceType: r.object.sourceType,
          packId: r.object.packId,
          packName: _packIdToName[r.object.packId] ??
              _humanizePackId(r.object.packId),
          sectionPath: r.object.sectionPath,
          score: r.score,
        ),
    ];
  }

  /// Ensure the BM25 index is built. Reads all chunks from ObjectBox and
  /// hands them to [Bm25Search.buildIndex]. Guarded by [_bm25BuildFuture]
  /// to prevent concurrent builds.
  Future<void> _ensureBm25Index() async {
    if (_bm25.isBuilt) return;
    if (_bm25BuildFuture != null) {
      await _bm25BuildFuture;
      return;
    }
    _bm25BuildFuture = _doBuildBm25Index();
    try {
      await _bm25BuildFuture;
    } finally {
      _bm25BuildFuture = null;
    }
  }

  Future<void> _doBuildBm25Index() async {
    await _ensureDb();
    if (_box == null || _box!.count() == 0) return;

    final all = _box!.getAll();
    _bm25.buildIndex([
      for (final c in all) (chunkId: c.chunkId, text: c.text, packId: c.packId),
    ]);
    debugPrint('[ash] BM25 index built: ${all.length} documents');
  }

  /// Keyword search via BM25. Returns hydrated [RetrievedInfo] list sorted
  /// by BM25 score descending (higher = better match).
  Future<List<RetrievedInfo>> _retrieveBM25(
    String queryText,
    int topK,
  ) async {
    await _ensureBm25Index();
    if (!_bm25.isBuilt) return [];

    final hits = _bm25.search(queryText, topK: topK, packIds: _lensPackIds);
    if (hits.isEmpty) return [];

    // Hydrate chunkIds → full ChunkEntity in a single ObjectBox query.
    final ids = hits.map((h) => h.chunkId).toList();
    final qb = _box!.query(ChunkEntity_.chunkId.oneOf(ids)).build();
    final entities = {for (final e in qb.find()) e.chunkId: e};
    qb.close();

    // Preserve BM25 ranking order and attach scores.
    return [
      for (final h in hits)
        if (entities.containsKey(h.chunkId))
          RetrievedInfo(
            chunkId: h.chunkId,
            text: entities[h.chunkId]!.text,
            source: entities[h.chunkId]!.source,
            sourceType: entities[h.chunkId]!.sourceType,
            packId: entities[h.chunkId]!.packId,
            packName: _packIdToName[entities[h.chunkId]!.packId] ??
                _humanizePackId(entities[h.chunkId]!.packId),
            sectionPath: entities[h.chunkId]!.sectionPath,
            score: h.score,
          ),
    ];
  }

  /// Merge vector and BM25 result lists using Reciprocal Rank Fusion.
  ///
  /// Each chunk's RRF score is `sum(1/(k + rank))` from each list where it
  /// appears. Results are sorted descending by RRF score (higher = better).
  List<RetrievedInfo> _fuseWithRrf(
    List<RetrievedInfo> vectorResults,
    List<RetrievedInfo> bm25Results,
    int topK,
  ) {
    // Build chunkId → rank maps (1-based).
    final vectorRanks = <String, int>{};
    for (var i = 0; i < vectorResults.length; i++) {
      vectorRanks[vectorResults[i].chunkId] = i + 1;
    }
    final bm25Ranks = <String, int>{};
    for (var i = 0; i < bm25Results.length; i++) {
      bm25Ranks[bm25Results[i].chunkId] = i + 1;
    }

    // Collect all unique chunk IDs.
    final allIds = <String>{...vectorRanks.keys, ...bm25Ranks.keys};

    // Build lookup by chunkId for both result lists.
    final byId = <String, RetrievedInfo>{};
    for (final r in vectorResults) {
      byId[r.chunkId] = r;
    }
    for (final r in bm25Results) {
      // BM25 result may carry a better-matched chunk the vector search missed.
      byId.putIfAbsent(r.chunkId, () => r);
    }

    // Compute RRF scores.
    final scored = <({String chunkId, double rrf})>[];
    for (final id in allIds) {
      var rrf = 0.0;
      if (vectorRanks.containsKey(id)) {
        rrf += 1.0 / (_rrfK + vectorRanks[id]!);
      }
      if (bm25Ranks.containsKey(id)) {
        rrf += 1.0 / (_rrfK + bm25Ranks[id]!);
      }
      scored.add((chunkId: id, rrf: rrf));
    }

    // Sort descending by RRF score.
    scored.sort((a, b) => b.rrf.compareTo(a.rrf));

    // Take top-K and return RetrievedInfo with RRF score.
    return [
      for (final s in scored.take(topK))
        RetrievedInfo(
          chunkId: s.chunkId,
          text: byId[s.chunkId]!.text,
          source: byId[s.chunkId]!.source,
          sourceType: byId[s.chunkId]!.sourceType,
          packId: byId[s.chunkId]!.packId,
          packName: byId[s.chunkId]!.packName,
          sectionPath: byId[s.chunkId]!.sectionPath,
          score: s.rrf,
        ),
    ];
  }

  /// Hybrid retrieval: vector (cosine) + BM25 (keyword) → RRF fusion.
  ///
  /// Returns the fused result list and the best cosine distance from the
  /// vector pass (used by the adaptive rewrite trigger). The fused list's
  /// scores are RRF scores (higher = better), while [bestVectorScore] is
  /// cosine distance (lower = better).
  Future<({List<RetrievedInfo> chunks, double? bestVectorScore})>
      _retrieveHybrid(
    List<double> queryEmbedding,
    String queryText,
    int topK,
  ) async {
    // Vector pass — threshold-filtered cosine results.
    final vectorResults = await _retrieveVector(queryEmbedding, _retrieveTopK);
    final bestVectorScore =
        vectorResults.isEmpty ? null : vectorResults.first.score;

    // BM25 pass — keyword results.
    final bm25Results = await _retrieveBM25(queryText, _bm25CandidateK);
    debugPrint('[ash] hybrid retrieval: ${vectorResults.length} vector, '
        '${bm25Results.length} BM25 candidates');

    // Fuse and return.
    final fused = _fuseWithRrf(vectorResults, bm25Results, topK);
    return (chunks: fused, bestVectorScore: bestVectorScore);
  }

  /// Fallback display name when we don't have a cached packName for a
  /// retrieved chunk's packId (e.g. the pack was imported in a prior session
  /// and the cache hasn't been warmed yet on this launch). Converts
  /// "dirty-drinking-water" → "Dirty Drinking Water".
  String _humanizePackId(String packId) {
    if (packId.isEmpty) return '';
    return packId
        .split(RegExp(r'[-_]'))
        .map((w) => w.isEmpty ? w : '${w[0].toUpperCase()}${w.substring(1)}')
        .join(' ');
  }

  // --- System instruction + context assembly ---

  /// Build the system instruction. Four variants, in precedence order:
  ///   1. image + live → short visual prose (voice-mode + vision combo)
  ///   2. image alone → focused visual-task prompt; NO RAG framing or
  ///      citation directive (those harm Gemma's attention on the photo)
  ///   3. live voice mode → tuned for TTS playback: plain prose, no
  ///      markdown, no emoji, 1-3 sentences. The user hears the reply,
  ///      not reads it, so headers/lists/`**bold**` are all hostile.
  ///   4. text + RAG on → general + "cite as [N] when reference material
  ///      is provided"
  ///   5. text + RAG off → general (no emergency framing)
  String _buildSystemPrompt({
    required bool useRag,
    bool liveMode = false,
    bool usesImage = false,
  }) {
    if (usesImage && liveMode) return _liveVisionSystemPrompt;
    if (usesImage) return _visionSystemPrompt;
    if (liveMode) return _liveVoiceSystemPrompt;
    if (!useRag) return _generalSystemPrompt;
    return _ragAwareSystemPrompt;
  }

  static const _generalSystemPrompt =
      'You are Ash, a calm offline assistant. Be brief, warm, and '
      'actionable. Answer from your general knowledge in plain '
      'conversational language. Do not assume the user is in an emergency.';

  static const _ragAwareSystemPrompt =
      'You are Ash, a calm offline assistant built for survival and '
      'emergency situations. Be brief, warm, and actionable. '
      'When reference material is provided ahead of the user\'s message '
      'in labeled blocks like [1], [2], ground your answer in it and cite '
      'the specific blocks you used inline as [1] or [2]. When no '
      'reference material is provided, give your best practical guidance '
      'from general knowledge. '
      'When the user describes an injury or emergency, walk them through '
      'clear immediate steps — do NOT refuse with "I\'m just an AI" or '
      '"I can\'t provide medical help". The user installed this app '
      'because they need help right now. Always recommend calling '
      'emergency services AFTER giving them something they can do this '
      'minute, never as a substitute.';

  /// Image queries. Drop the RAG/citation framing entirely — those
  /// instructions distract the model from the visual content and produce
  /// answers like "Based on [1] from the Earthquake guide..." even when
  /// no reference was injected. Keep this prompt short and visual-first.
  static const _visionSystemPrompt =
      'You are Ash, a calm offline assistant. The user has shared an image. '
      'Look at the image carefully and answer their question based on what '
      'you actually see. Be specific about details in the image — colors, '
      'objects, people, text, hazards. Keep your answer focused and '
      'practical. If the image doesn\'t contain enough information to '
      'answer, say so honestly.';

  /// Image queries + live voice mode. Same visual focus, but reply must
  /// be TTS-friendly (short prose, no markdown).
  static const _liveVisionSystemPrompt =
      'You are Ash, speaking aloud about an image the user shared. Look at '
      'the image carefully. Describe what you see and answer the user\'s '
      'question in 1–3 sentences. Plain spoken prose only — no markdown, '
      'no emoji, no lists. Be specific about what\'s actually in the image.';

  /// Live voice mode: the user hears the reply read aloud. Every markdown
  /// marker (`**bold**`, `### heading`, bullets, numbered lists) becomes
  /// awkward narration ("asterisk asterisk bold asterisk asterisk").
  /// Emoji read literally too ("face with tears of joy"). Force the model
  /// into plain conversational prose, kept short.
  static const _liveVoiceSystemPrompt =
      'You are Ash, speaking aloud to the user through a voice interface. '
      'The user hears your reply, they do not read it. Therefore: '
      '(1) Keep replies to 1–3 sentences unless the user explicitly asks '
      'for detail. '
      '(2) No markdown — never use asterisks, hashes, bullets, numbered '
      'lists, or headers. '
      '(3) No emoji. '
      '(4) No citation brackets like [1] — speak naturally, mention the '
      'source by name only if essential. '
      '(5) Use plain conversational prose, like talking to a friend. '
      '(6) For safety procedures, speak them as short imperative '
      'sentences, not bullet points.';

  /// Per-chunk character cap. With round-5 proposition chunks (one full
  /// Q&A pair per chunk), 1200 chars ≈ 320 tokens — enough to inject most
  /// answers verbatim without truncation. Long procedural answers (CPR
  /// steps, tourniquet construction) may still get trimmed at the
  /// sentence boundary, which is acceptable.
  static const _maxChunkChars = 1200;

  /// Result of packing retrieved chunks into a prompt-ready context block.
  /// Carries both the assembled string (for prompt injection) and the list
  /// of [MessageSource] entries that actually made it in — guaranteed to use
  /// the same `[N]` numbering as the embedded labels so the UI can render
  /// matching citation chips.
  ///
  /// Round 3: each chunk runs through [selectRelevantContent] before being
  /// emitted. The HazAdapt chunker treats `(phase, audience)` as the atomic
  /// unit, so a single chunk can be 4–6k chars covering multiple sub-topics
  /// (e.g. "Active Shooter > Response > RUN.HIDE.FIGHT." bundles RUN, HIDE,
  /// and FIGHT into one chunk). The selector splits at internal `####`
  /// headings and picks the sub-block(s) that best match the user's query,
  /// so the model sees focused context instead of a 4k-char wall trimmed
  /// arbitrarily at char 800.
  ({String text, List<MessageSource> sources}) _assembleContext(
    List<RetrievedInfo> chunks,
    String userQuery,
  ) {
    if (chunks.isEmpty) return (text: '', sources: const []);
    final sb = StringBuffer()
      ..writeln('Reference material (use if relevant; cite as [1], [2] '
          'inline):');
    final sources = <MessageSource>[];
    var used = 0;
    var blockNum = 0;
    for (final chunk in chunks) {
      final cleaned = sanitizeChunkText(chunk.text);
      if (cleaned.isEmpty) continue;

      // Pick the most-relevant slice of the chunk for the user's query.
      // For small chunks (<= _maxChunkChars) this is essentially a no-op;
      // for big multi-topic chunks it picks the on-topic sub-block(s).
      final text = selectRelevantContent(
        chunkText: cleaned,
        query: userQuery,
        maxChars: _maxChunkChars,
      );

      // Total-budget cap: don't start a new chunk if we'd blow past it.
      // Allow the first chunk through even if oversized so we always inject
      // at least one block when retrieval clears the threshold.
      if (blockNum > 0 && used + text.length > _contextCharBudget) break;

      blockNum++;
      final label = _formatSourceLabel(chunk, body: text);
      sb
        ..writeln()
        ..writeln('[$blockNum] $label')
        ..writeln(text);
      used += text.length;

      sources.add(MessageSource(
        id: blockNum,
        chunkId: chunk.chunkId,
        packId: chunk.packId,
        packName: chunk.packName,
        sectionPath: chunk.sectionPath,
        text: text,
        score: chunk.score,
      ));
    }
    if (blockNum == 0) return (text: '', sources: const []);
    return (text: sb.toString(), sources: sources);
  }

  /// Compose a short human label for a retrieved chunk's source block. The
  /// chunk's actual title lives inside its markdown body (e.g. "RUN" or
  /// "Immediate Emergency Actions") — much more meaningful than the
  /// HazAdapt taxonomy crumb "Response > For Adults". Prefer the extracted
  /// heading when we have one; fall back to the humanized section path.
  ///
  /// [body] is the sub-block actually being injected (post sub-block
  /// selection). It's the right source for the heading extraction because
  /// it reflects what the model is reading.
  String _formatSourceLabel(RetrievedInfo chunk, {required String body}) {
    final name = chunk.packName.isNotEmpty
        ? chunk.packName
        : (chunk.packId.isNotEmpty
            ? _humanizePackId(chunk.packId)
            : 'Reference');
    final extracted = extractChunkHeading(body);
    if (extracted != null && extracted.isNotEmpty) {
      return '$name — $extracted';
    }
    final section = humanizeSectionPath(chunk.sectionPath);
    if (section.isEmpty) return name;
    return '$name — $section';
  }

  /// Drop special vocab tokens that occasionally leak through the SDK's
  /// streaming output (`<unk>`, `<bos>`, `<eos>`, `<pad>`, `<unused*>`,
  /// `<start_of_turn>`, `<end_of_turn>`, etc). Returns true if the token is
  /// real text the user should see.
  static final _specialTokenRegex = RegExp(
    r'^<(?:unk|bos|eos|pad|unused\d+|start_of_turn|end_of_turn|s|/s)>$',
    caseSensitive: false,
  );
  bool _isUserVisibleToken(String token) {
    if (token.isEmpty) return false;
    return !_specialTokenRegex.hasMatch(token.trim());
  }

  // --- LLM helpers ---

  Future<void> _activateInstalledModel() async {
    // Always call the install builder. It's idempotent on disk (re-downloads
    // skipped when the file already exists) but it ALWAYS calls
    // setActiveModel(spec). On a fresh app launch the file is on disk but
    // no active spec is registered yet, so the early-return shortcut left
    // FlutterGemma.getActiveModel() throwing "No active inference model set".
    // Matches the gemma4_bench pattern.
    await FlutterGemma.installModel(
      modelType: ModelType.gemma4,
      fileType: ModelFileType.litertlm,
    ).fromNetwork(_activeModel.url).install();
  }

  String? _blankToNull(String? value) {
    final trimmed = value?.trim();
    return trimmed == null || trimmed.isEmpty ? null : trimmed;
  }
}
