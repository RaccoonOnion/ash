import 'dart:async';
import 'dart:typed_data';

import 'package:flutter_gemma/flutter_gemma.dart' show CancelToken;

import 'inference_settings.dart';
import 'llm_model.dart';

// Re-export so callers can `import 'inference_service.dart'` and get the
// CancelToken without importing flutter_gemma directly. Lets the
// pause/resume control surface stay model-agnostic at the call sites.
export 'package:flutter_gemma/flutter_gemma.dart' show CancelToken;

/// Where to run the model's matmul-heavy work. iOS .litertlm supports both;
/// GPU is faster and runs cooler, CPU is the fallback when GPU drivers
/// misbehave or the user wants to compare. Switching backends requires a
/// full engine reload (~5s text-only, ~30s for vision).
enum AcceleratorChoice { gpu, cpu }

/// One prior turn to seed a fresh chat session with — used on chat re-entry
/// (after the engine was closed for thermal reasons) so the model "remembers"
/// what was said before. Order matters: caller supplies oldest-first,
/// alternating user / assistant.
class HistoryTurn {
  const HistoryTurn({
    required this.isUser,
    required this.text,
    this.imageBytes,
  });
  final bool isUser;
  final String text;
  final Uint8List? imageBytes;
}

class ModelStatus {
  const ModelStatus({required this.isInstalled, this.error});
  final bool isInstalled;
  final String? error;
}

/// A reference chunk that grounded the assistant's reply. The streaming
/// service emits these (as a one-shot list on a preflight [InferenceChunk]
/// just before token streaming starts) so the UI can render citation chips
/// under the assistant bubble. The [id] matches the inline `[N]` marker the
/// model is asked to use when citing this block.
class MessageSource {
  const MessageSource({
    required this.id,
    required this.chunkId,
    required this.packId,
    required this.packName,
    required this.sectionPath,
    required this.text,
    required this.score,
  });

  /// 1-based index. Matches the `[N]` marker used by the model and the
  /// label shown on the citation chip.
  final int id;

  /// Stable identifier of the underlying chunk inside its pack. Used by the
  /// Library reader to scroll directly to the cited proposition rather than
  /// dumping the user at the top of the pack. Empty when the chunk was
  /// retrieved before this field existed (chats persisted from an older
  /// build).
  final String chunkId;

  /// Pack the chunk came from (e.g. `"cpr"`). Used by "Open in Library".
  final String packId;

  /// Display name of the pack (e.g. `"CPR"`).
  final String packName;

  /// Section breadcrumb inside the pack (e.g. `"During (React) > For Adults"`),
  /// when the pack provided one. Otherwise null.
  final String? sectionPath;

  /// Sanitized chunk text (the same text that was injected into the prompt).
  /// Shown to the user when they tap the chip.
  final String text;

  /// Cosine distance from the query embedding. Lower is a closer match.
  /// Surface as a soft confidence hint in the chip detail sheet.
  final double score;
}

class InferenceChunk {
  const InferenceChunk({
    required this.text,
    required this.isDone,
    this.sources,
  });
  final String text;
  final bool isDone;

  /// Set on a single preflight chunk emitted before token streaming starts —
  /// carries the references that grounded the reply (if RAG retrieved any
  /// that cleared the threshold). Null on every other chunk in the stream.
  final List<MessageSource>? sources;
}

/// What a hypothetical next query would force the service to do *before* it
/// can stream the reply. Used by the UI to surface a banner so the user knows
/// why the first token is slow (engine reload, vision swap, session rebuild
/// with history replay) instead of staring at a frozen "thinking…" dot.
class ReconfigureNeeds {
  const ReconfigureNeeds({
    this.engineReload = false,
    this.visionLoad = false,
    this.sessionRebuild = false,
  });

  /// maxTokens or accelerator change → full engine close + rebuild
  /// (~5s text, ~30s vision).
  final bool engineReload;

  /// First image query against a text-only engine → vision encoder load
  /// (~30s the first time; sticky thereafter).
  final bool visionLoad;

  /// chatId switch OR samplingSignature (temp/topK/topP/RAG) changed → close
  /// the persisted chat session and create a new one. History is replayed
  /// into the new session, so cost scales with chat length (~0.5–3s).
  final bool sessionRebuild;

  bool get isAnything => engineReload || visionLoad || sessionRebuild;
}

abstract interface class InferenceService {
  /// The LLM variant currently selected as active. The resident engine — if
  /// any — was built from this variant's `.litertlm`. Switching variants is
  /// done via [setActiveLlmModel].
  LlmModel get activeLlmModel;

  /// Replace the active variant. Closes the resident engine; the next query
  /// (or explicit [warmUp]) re-creates one from the new variant's file. The
  /// new variant must already be installed — call [installModel] first if
  /// it isn't.
  Future<void> setActiveLlmModel(LlmModel model);

  /// Whether [model] (or [activeLlmModel] if omitted) is downloaded and
  /// available on disk. The set of installed variants is independent of the
  /// active one — multiple may coexist; only one is loaded at a time.
  Future<ModelStatus> getModelStatus({LlmModel? model});

  /// Download [model] (or [activeLlmModel] if omitted) from HuggingFace. The
  /// underlying SDK is idempotent: re-installing an already-present file
  /// skips the network fetch but still refreshes the active-model registry.
  ///
  /// Pause / resume / cancel are wired through [cancelToken] — call
  /// `cancelToken.cancel('paused')` to interrupt mid-download. The partial
  /// file stays on disk; calling [installModel] again with a fresh token
  /// resumes from where it left off (HuggingFace `resume_download` is on
  /// by default). On cancel the returned Future completes with a
  /// `DownloadCancelledException` that callers should check via
  /// `CancelToken.isCancel(error)` before treating it as a real failure.
  Future<void> installModel({
    LlmModel? model,
    String? token,
    void Function(double progress)? onProgress,
    CancelToken? cancelToken,
  });

  /// Remove [model] (or [activeLlmModel] if omitted) from disk. Useful when
  /// the user wants to reclaim space after switching variants.
  Future<void> deleteModel({LlmModel? model});

  /// The accelerator currently in effect. Reflects what the resident engine
  /// was built with; [setAccelerator] swaps it.
  AcceleratorChoice get accelerator;

  /// Replace the active accelerator. If an engine is already resident, it's
  /// closed and (asynchronously) re-warmed with the new backend.
  Future<void> setAccelerator(AcceleratorChoice choice);

  /// Whether speculative decoding (Multi-Token Prediction) is enabled on
  /// the engine. Only meaningful for `.litertlm` files that ship an MTP
  /// drafter — Gemma 4 E2B-it and E4B-it both do as of LiteRT-LM v0.11.
  /// Older files silently ignore the flag at the SDK level. Default on
  /// because it's a ~1.5–2x decode speedup when supported.
  bool get speculativeDecoding;

  /// Toggle MTP. Like [setAccelerator], this is an engine-level setting:
  /// changing it closes the resident engine and re-warms with the new
  /// flag.
  Future<void> setSpeculativeDecoding(bool enabled);

  /// Warms up the on-device model in text-only mode so the user can start
  /// chatting quickly without paying the vision encoder load cost. Safe to
  /// call multiple times — returns the same in-flight future or no-ops if
  /// already warm.
  Future<void> warmUp();

  /// Whether the model is loaded and ready to answer a text query without a
  /// further cold load. True after [warmUp] completes (text-only) or after a
  /// vision swap.
  bool get isWarm;

  /// Whether the currently loaded engine has vision enabled. Image queries
  /// against a text-only engine trigger an internal swap to a vision engine.
  bool get hasVision;

  /// Whether an image-attached query is currently waiting for a text → vision
  /// engine swap (a ~30s cold load on first ever vision use, faster after).
  bool get isLoadingVision;

  /// Releases the underlying engine. Use this when the user leaves the chat
  /// surface to drop GPU/Metal resources and let the device cool. The next
  /// query (or explicit [warmUp]) re-creates the engine.
  Future<void> closeEngine();

  /// Drop the persisted chat session for the current chat without
  /// releasing the engine. Use after trimming displayed messages so the
  /// next query rebuilds the session against the now-shorter history.
  /// Cheap relative to [closeEngine] (keeps the model warm), but the
  /// first turn after a reset re-replays history into the KV cache.
  Future<void> resetChatSession();

  /// Run a single chat turn for [chatId]. Successive calls with the same
  /// [chatId] reuse the underlying [InferenceChat] so the model sees prior
  /// turns as conversation history. A different [chatId] closes the prior
  /// chat session and starts a fresh one. Use [closeEngine] to release the
  /// chat (and engine) when leaving the chat surface.
  ///
  /// [priorHistory] is replayed into the model's KV cache ONLY when the
  /// service has to create a new chat session — this happens on first query
  /// for a chat, after a settings change, or after re-entering a chat whose
  /// engine was closed by [closeEngine]. Subsequent same-chat turns ignore
  /// [priorHistory] and rely on the persisted session.
  Stream<InferenceChunk> query({
    required String chatId,
    required String prompt,
    Uint8List? imageBytes,
    List<HistoryTurn> priorHistory = const [],
    InferenceSettings settings = InferenceSettings.defaults,
    /// When true, the service swaps in a voice-tuned system prompt that
    /// tells the model to reply in plain prose: no markdown, no emoji,
    /// no numbered lists, 1-3 sentences. Used by [LiveVoiceScreen] so the
    /// TTS playback doesn't read "asterisk asterisk bold asterisk asterisk".
    bool liveMode = false,
  });
  Future<void> stop();

  /// True only while a `query()` stream is actively producing tokens.
  /// Callers use this to avoid invoking [stop] when generation has
  /// already finished — calling `stopGeneration` on a completed chat
  /// session on iOS LiteRT-LM has been observed to segfault (which
  /// Dart's `try/catch` cannot trap because it happens in native code).
  bool get isGenerating;

  /// Predict what the next [query] call with these args will have to rebuild
  /// before it can stream tokens. Pure read of internal state — does not
  /// mutate anything. Returns flags the UI can use to render a status banner.
  ReconfigureNeeds reconfigureNeedsFor({
    required String chatId,
    required InferenceSettings settings,
    required bool needsVision,
  });

  /// Eagerly apply [settings] to the resident engine + chat session for
  /// [chatId] without sending a prompt. Use this when the user closes the
  /// settings sheet so the work (engine reload, session rebuild + history
  /// replay) happens up front and the next [query] streams instantly. Safe
  /// to call when already current — no-ops if nothing needs rebuilding.
  ///
  /// [priorHistory] is replayed only when the chat session has to be
  /// rebuilt (same rules as [query]). Caller is responsible for not invoking
  /// this while a [query] stream is still in flight against the same chat.
  Future<void> applySettings({
    required String chatId,
    required InferenceSettings settings,
    List<HistoryTurn> priorHistory = const [],
  });

  /// Import a knowledge pack from bundled assets into the vector DB.
  Future<void> importPack(String packId);

  /// Download a knowledge pack from a remote URL and import it. Used by
  /// packs marked `PackSource.remote` in the registry. Progress callback
  /// receives values in [0..1]; `1.0` lands once chunks are committed.
  ///
  /// SHA-256 verification will land alongside the CDN rollout — see the
  /// TODO in the implementation. Until then the URL itself is the trust
  /// boundary.
  Future<void> importPackFromUrl(
    String packId,
    String url, {
    void Function(double progress)? onProgress,
  });

  /// Remove all chunks for a pack from the vector DB.
  Future<void> uninstallPack(String packId);

  /// Scope retrieval for subsequent queries to a set of packs (by packId).
  /// Pass `null` (or an empty set) to retrieve across every installed pack —
  /// this is the default for fresh chats. A non-empty set restricts the
  /// HNSW search to just those packs, e.g. for "Ask Ash about this section"
  /// flows that pin a chat to one pack.
  void setLens(Set<String>? lensPackIds);

  /// Import all packs that are bundled as assets.
  ///
  /// (Round 2: this method is no longer called from launch — the host only
  /// imports essentials by default, and the rest become real install actions
  /// the user takes from the Store. Kept for callers who want a one-shot
  /// "fill the library" hook.)
  Future<void> importAllPacks();

  /// Set of packIds that have chunks in the vector store. Used at launch to
  /// figure out which packs are "installed" (carried over from a previous
  /// session) so the Library / Store split reflects reality.
  Future<Set<String>> installedPackIds();

  /// Read every chunk of [packId] back from the vector store as plain
  /// reference material — used by the Library reader screen. The chunks are
  /// returned in insertion order so they read sequentially. Sanitization
  /// is applied so the reader shows the same cleaned text the model sees.
  Future<List<LibraryChunk>> readPack(String packId);

  /// Rank a set of packs by semantic similarity to [query]. Reuses the
  /// retrieval-time MiniLM embeddings — no second model required. Per-pack
  /// embeddings are cached internally by packId. Results are sorted by
  /// cosine similarity, descending.
  ///
  /// [texts] is the corpus to rank against, keyed by packId. The value is
  /// the human-language text the embedding should represent (typically
  /// `"{name}\n{summary}"`).
  Future<List<PackRanking>> rankByQuery({
    required String query,
    required Map<String, String> texts,
  });
}

/// One row of [InferenceService.rankByQuery]'s result. [score] is cosine
/// similarity in roughly [-1, 1] for L2-normalized MiniLM embeddings —
/// callers can threshold-filter (e.g. only show recommendations above 0.3).
class PackRanking {
  const PackRanking({required this.packId, required this.score});
  final String packId;
  final double score;
}

/// One chunk of reference material for the Library reader. Mirrors what's
/// in the vector store minus the embedding (the reader doesn't need it).
class LibraryChunk {
  const LibraryChunk({
    required this.chunkId,
    required this.text,
    required this.sectionPath,
  });

  final String chunkId;
  final String text;
  final String? sectionPath;
}
