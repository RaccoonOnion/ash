/// Per-conversation, user-tunable inference knobs.
///
/// Sampling params (temperature/topK/topP) are baked into the chat session at
/// creation; changing them invalidates the persisted chat. [maxTokens] is
/// baked into the underlying *engine* (KV-cache size) — changing it forces a
/// full engine reload on the next query. [useRag] is checked per query, no
/// session/engine impact.
///
/// Defaults: RAG on (so survival-pack knowledge is grounding replies out of
/// the box — this is the app's whole point), and the Google AI Edge Gallery
/// preset for Gemma 4 E2B-it sampling — temperature 1.0, top-K 64, top-P
/// 0.95, max tokens 4000 (Google's defaultMaxToken for this model). Image-
/// bearing queries auto-bump maxTokens to ≥4096 inside the service since
/// vision patches eat context.
class InferenceSettings {
  const InferenceSettings({
    this.useRag = true,
    this.temperature = 1.0,
    this.topK = 64,
    this.topP = 0.95,
    this.maxTokens = 4000,
    this.isThinking = false,
  });

  static const defaults = InferenceSettings();

  /// Whether to retrieve and inject RAG context for text queries. Image
  /// queries skip RAG unconditionally.
  final bool useRag;

  /// Sampling temperature (0 = greedy, 1+ = increasingly random).
  final double temperature;

  /// Top-K sampling: only consider the K most likely next tokens.
  final int topK;

  /// Nucleus (top-P) sampling: only consider the smallest set of tokens whose
  /// cumulative probability exceeds P.
  final double topP;

  /// Engine KV-cache size (matches Google AI Edge Gallery's "Max tokens" —
  /// covers system instruction + history + RAG context + user prompt + the
  /// model's reply, all sharing this budget). Bigger = more conversation
  /// memory but more RAM and slower engine init.
  ///
  /// Changing this value requires an engine reload (~5s text, ~30s for the
  /// first vision swap of a session).
  final int maxTokens;

  /// Gemma 4 thinking mode. When true, the model generates an internal
  /// reasoning trace inside `<think>...</think>` tags before the final
  /// answer; the runtime stream surfaces the trace as `ThinkingResponse`
  /// events the UI can render separately. Costs latency proportional to
  /// the thinking budget — turn off for casual chat, on for harder
  /// reasoning. Baked into the chat session at create-time, so toggling
  /// invalidates the persisted chat.
  final bool isThinking;

  /// Signature used by the service to detect session-affecting changes and
  /// rebuild the persisted chat session. Includes [useRag] because the
  /// system prompt baked into the session changes when RAG is toggled,
  /// and [isThinking] because it's a createChat-level flag. Excludes
  /// [maxTokens] — that's engine-level and tracked separately.
  String get samplingSignature =>
      '${temperature.toStringAsFixed(3)}|$topK|${topP.toStringAsFixed(3)}|'
      'rag=$useRag|think=$isThinking';

  InferenceSettings copyWith({
    bool? useRag,
    double? temperature,
    int? topK,
    double? topP,
    int? maxTokens,
    bool? isThinking,
  }) {
    return InferenceSettings(
      useRag: useRag ?? this.useRag,
      temperature: temperature ?? this.temperature,
      topK: topK ?? this.topK,
      topP: topP ?? this.topP,
      maxTokens: maxTokens ?? this.maxTokens,
      isThinking: isThinking ?? this.isThinking,
    );
  }
}
