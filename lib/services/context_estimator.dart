import '../models/chat.dart';

/// Char-based context-load estimator. Returns a fraction 0..N where 1.0
/// means the projected prompt is right at the engine's window; >1.0 means
/// the model will silently truncate (KV-cache eviction) or hang.
///
/// We deliberately don't tokenize on-device. A 4-chars-per-token proxy is
/// conservative enough for an English-only assistant ("are we 80% full?"
/// is the question this answers, not "exactly how many tokens"), costs
/// nothing to compute on each rebuild, and stays stable across keystrokes.
///
/// Components:
///   - all prior message bodies (user + assistant), plus the live draft
///   - a fixed system-prompt + ground-rules overhead
///   - the RAG context budget when retrieval is on (the service caps
///     retrieved context at [_ragContextChars])
///
/// The 4 chars/token ratio errs on the side of overcounting: tokens for
/// proper English prose typically run 3.5-4.0 chars, but JSON / code /
/// mid-stream UTF-8 punctuation pull it down. We'd rather warn early than
/// surprise the user with a silent truncation.
const _charsPerToken = 4.0;
const _systemOverheadTokens = 500;
const _ragContextChars = 3000;

class ContextLoadEstimate {
  const ContextLoadEstimate({
    required this.usedTokens,
    required this.maxTokens,
  });

  final int usedTokens;
  final int maxTokens;

  double get fraction => maxTokens == 0 ? 0 : usedTokens / maxTokens;
  int get pct => (fraction * 100).round();
}

ContextLoadEstimate estimateContextLoad(Chat chat, String draft) {
  var chars = 0;
  for (final m in chat.messages) {
    chars += m.text.length;
  }
  chars += draft.length;
  var tokens = (chars / _charsPerToken).round() + _systemOverheadTokens;
  if (chat.inferenceSettings.useRag) {
    tokens += (_ragContextChars / _charsPerToken).round();
  }
  return ContextLoadEstimate(
    usedTokens: tokens,
    maxTokens: chat.inferenceSettings.maxTokens,
  );
}

/// Drop the oldest [fraction] of the chat's messages in place. Used by
/// the "Trim older messages" action on the context-warning banner. The
/// chat session inside the inference service should be reset right after
/// (so the model's KV cache also forgets the trimmed turns); this helper
/// only touches the displayed transcript.
///
/// Always rounds DOWN to a turn boundary (pair of user + assistant), so
/// the visible conversation never starts on a dangling assistant reply
/// with no preceding user turn.
int trimOldestMessages(Chat chat, {double fraction = 0.3}) {
  final n = chat.messages.length;
  if (n < 4) return 0; // Need at least 2 full turns to bother trimming.
  var dropCount = (n * fraction).floor();
  if (dropCount < 2) dropCount = 2;
  // Snap to even count so we don't end on a half-turn.
  if (dropCount.isOdd) dropCount -= 1;
  if (dropCount <= 0) return 0;
  if (dropCount >= n - 2) dropCount = n - 2; // always keep at least one turn
  chat.messages.removeRange(0, dropCount);
  return dropCount;
}
