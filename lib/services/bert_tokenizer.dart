/// Minimal BERT WordPiece tokenizer for all-MiniLM-L6-v2.
///
/// Loads a vocab.txt file (one token per line) and performs:
/// 1. Basic tokenization (lowercase, whitespace/punctuation split)
/// 2. WordPiece subword tokenization
/// 3. Adds [CLS] / [SEP] tokens
class BertTokenizer {
  BertTokenizer(List<String> vocabLines)
      : _vocab = {
          for (var i = 0; i < vocabLines.length; i++)
            vocabLines[i].trim(): i,
        },
        _unkId = _findUnkId(vocabLines);

  final Map<String, int> _vocab;
  final int _unkId;

  static int _findUnkId(List<String> lines) {
    for (var i = 0; i < lines.length; i++) {
      if (lines[i].trim() == '[UNK]') return i;
    }
    return 0;
  }

  /// Tokenize [text] and return token IDs with attention mask.
  ({List<int> inputIds, List<int> attentionMask}) tokenize(
    String text, {
    int maxLength = 128,
  }) {
    final basicTokens = _basicTokenize(text.toLowerCase());
    final subwordTokens = <String>['[CLS]'];
    for (final token in basicTokens) {
      if (token.isEmpty) continue;
      subwordTokens.addAll(_wordpiece(token));
    }
    subwordTokens.add('[SEP]');

    // Truncate to maxLength
    if (subwordTokens.length > maxLength) {
      subwordTokens.removeRange(maxLength - 1, subwordTokens.length);
      subwordTokens.add('[SEP]');
    }

    final inputIds = subwordTokens
        .map((t) => _vocab[t] ?? _unkId)
        .toList();

    // Pad to maxLength
    while (inputIds.length < maxLength) {
      inputIds.add(0);
    }

    final attentionMask = List<int>.generate(maxLength, (i) {
      final token = i < subwordTokens.length ? subwordTokens[i] : '';
      return token == '' ? 0 : 1;
    });

    return (inputIds: inputIds, attentionMask: attentionMask);
  }

  List<String> _basicTokenize(String text) {
    // Strip accented chars, split punctuation, split whitespace
    final cleaned = text
        .replaceAll(
          RegExp(r'[^\w\s]'),
          ' ',
        )
        .replaceAll(RegExp(r'\s+'), ' ')
        .trim();
    if (cleaned.isEmpty) return [];
    return cleaned.split(' ');
  }

  List<String> _wordpiece(String word) {
    if (_vocab.containsKey(word)) return [word];

    final tokens = <String>[];
    var start = 0;
    while (start < word.length) {
      var end = word.length;
      var found = false;
      while (start < end) {
        final substr = start == 0
            ? word.substring(start, end)
            : '##${word.substring(start, end)}';
        if (_vocab.containsKey(substr)) {
          tokens.add(substr);
          found = true;
          break;
        }
        end--;
      }
      if (!found) {
        return ['[UNK]'];
      }
      start = end;
    }
    return tokens;
  }
}
