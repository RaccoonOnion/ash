/// Strip text that doesn't belong in TTS playback.
///
/// The chat assistant streams markdown-rich replies (bold, headers, bullets,
/// citation chips like `[1]`, emoji). Reading those literally aloud sounds
/// like "asterisk asterisk bold asterisk asterisk" or "fire emoji". The live
/// voice screen feeds each chunk of model output through this filter before
/// queueing it for TTS, so the user hears clean prose even when the model
/// occasionally slips into markdown despite the voice-mode system prompt.
///
/// Defensive: also runs even when the model is well-behaved, since edge
/// cases (a `[N]` citation marker, an accidental `**emphasis**`) can still
/// slip through.
/// Sanitize text for TTS playback.
///
/// [trimEnds] — when true (the default, for whole-utterance use cases),
/// strips leading/trailing whitespace from the result. **MUST be false**
/// when called per-streaming-chunk: Gemma's SentencePiece tokenizer emits
/// tokens like `" tourniquet"` with a leading space that marks the word
/// boundary. Trimming per-chunk strips those boundary spaces, which makes
/// `feedTtsChunk` glue adjacent tokens into one mangled word like
/// `"previousword"` — the "sub-word pronunciation" symptom.
String sanitizeForTts(String text, {bool trimEnds = true}) {
  if (text.isEmpty) return '';
  var s = text;

  // 0. Drop thinking blocks entirely. When the chat has thinking mode on,
  //    the model emits `<think>reasoning trace</think>final answer`. The
  //    TTS layer is for the *answer*, not the reasoning — without this
  //    strip the user hears "less than think greater than… less than slash
  //    think greater than" or the reasoning read aloud verbatim. Handles
  //    both fully-closed blocks and mid-stream chunks that opened `<think>`
  //    but haven't seen `</think>` yet.
  s = s.replaceAll(RegExp(r'<think>.*?</think>', dotAll: true), '');
  // Strip from an unclosed `<think>` to end-of-string for mid-stream
  // chunks — the closing tag and the answer will arrive in a later chunk.
  final openIdx = s.indexOf('<think>');
  if (openIdx >= 0 && !s.contains('</think>', openIdx)) {
    s = s.substring(0, openIdx);
  }
  // Also drop a leading `</think>` if the previous chunk ended mid-thought.
  s = s.replaceFirst(RegExp(r'^.*?</think>', dotAll: true), '');

  // 1. Strip emoji and pictographic symbols. Covers the common ranges
  //    without enumerating every Unicode block.
  s = _stripEmoji(s);

  // 2. Drop markdown emphasis markers but keep their content.
  //    `**bold**` → `bold`, `*italic*` → `italic`, `__under__` → `under`.
  //
  //    CRITICAL: must use `replaceAllMapped` (callback) — Dart's
  //    `replaceAll(Pattern, String)` does NOT interpret `$1` as a
  //    backreference. Passing `r'$1'` literally substitutes the
  //    two-character string `$1`, which AVSpeechSynthesizer then reads
  //    aloud as "one dollar" — and the original bold content is lost.
  //    Symptom: live mode would speak "one dollar Tap the person…"
  //    instead of "Check for Responsiveness: Tap the person…".
  s = s.replaceAllMapped(
    RegExp(r'\*{1,3}(.*?)\*{1,3}', dotAll: false),
    (m) => m.group(1) ?? '',
  );
  s = s.replaceAllMapped(
    RegExp(r'_{1,3}(.*?)_{1,3}', dotAll: false),
    (m) => m.group(1) ?? '',
  );

  // 3. Strip heading hashes — `#### Title` → `Title`.
  s = s.replaceAll(RegExp(r'^\s*#{1,6}\s+', multiLine: true), '');

  // 4. Strip blockquote markers — `> text` → `text`.
  s = s.replaceAll(RegExp(r'^\s*>\s?', multiLine: true), '');

  // 5. Drop bullet / numbered list markers at line start.
  //    `- item` → `item`, `* item` → `item`, `1. item` → `item`.
  s = s.replaceAll(RegExp(r'^\s*[-*+]\s+', multiLine: true), '');
  s = s.replaceAll(RegExp(r'^\s*\d+\.\s+', multiLine: true), '');

  // 6. Strip citation brackets like `[1]` / `[12]` — they're meaningless
  //    in audio. (Markdown links `[text](url)` were already collapsed at
  //    build time, but a stray `[N]` from the model's reply still gets
  //    here.)
  s = s.replaceAll(RegExp(r'\[\d+\]'), '');

  // 7. Strip inline / fenced code markers — keep the code text.
  s = s.replaceAll(RegExp(r'`{1,3}'), '');

  // 8. Collapse the whitespace that's now scattered through the string.
  s = s.replaceAll(RegExp(r'[ \t]+'), ' ');
  s = s.replaceAll(RegExp(r'\n{2,}'), '\n');

  // 9. Optionally trim at ends. Callers iterating over streaming chunks
  //    should pass trimEnds: false (see the function-level doc).
  return trimEnds ? s.trim() : s;
}

/// Strip emoji + pictographic codepoints. Walks runes once and skips any
/// codepoint that falls in a known emoji range. Faster than running ~20
/// regex passes for each block.
String _stripEmoji(String input) {
  final out = StringBuffer();
  for (final cu in input.runes) {
    if (_isEmoji(cu)) continue;
    out.writeCharCode(cu);
  }
  return out.toString();
}

bool _isEmoji(int cu) {
  // Variation selectors / zero-width joiners around emoji sequences.
  if (cu == 0xFE0F || cu == 0xFE0E || cu == 0x200D) return true;
  // Misc symbols + dingbats (☀ ☂ ✂ ✈ ✨ etc.)
  if (cu >= 0x2600 && cu <= 0x27BF) return true;
  // Various technical / shapes ranges that include some emoji.
  if (cu >= 0x2300 && cu <= 0x23FF) return true;
  // Enclosed alphanumerics supplement (🅰 🆎 etc.)
  if (cu >= 0x1F100 && cu <= 0x1F1FF) return true;
  // Miscellaneous symbols and pictographs (🌀 🌐 ☎ etc.)
  if (cu >= 0x1F300 && cu <= 0x1F5FF) return true;
  // Emoticons (😀 😂 etc.)
  if (cu >= 0x1F600 && cu <= 0x1F64F) return true;
  // Transport & map symbols (🚀 🚗 etc.)
  if (cu >= 0x1F680 && cu <= 0x1F6FF) return true;
  // Geometric shapes extended (🔴 🟢 etc.)
  if (cu >= 0x1F780 && cu <= 0x1F7FF) return true;
  // Supplemental symbols and pictographs (🤖 🤝 etc.)
  if (cu >= 0x1F900 && cu <= 0x1F9FF) return true;
  // Symbols and pictographs extended-A (🪐 🪜 etc.)
  if (cu >= 0x1FA70 && cu <= 0x1FAFF) return true;
  // Regional indicator symbols — country flags (🇺🇸 etc.)
  if (cu >= 0x1F1E6 && cu <= 0x1F1FF) return true;
  return false;
}
