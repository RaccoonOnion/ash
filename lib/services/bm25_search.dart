import 'dart:math';

/// Self-contained BM25 index and scorer. No external dependencies.
///
/// Built lazily from ObjectBox chunks, invalidated when packs are
/// imported/uninstalled. Uses standard Robertson–Sparck Jones parameter
/// defaults (k1=1.5, b=0.75).
class Bm25Search {
  static const _k1 = 1.5;
  static const _b = 0.75;

  /// ~30 common English stop words. Sufficient at this corpus size.
  static const _stopWords = {
    'a',
    'an',
    'the',
    'and',
    'or',
    'but',
    'in',
    'on',
    'at',
    'to',
    'for',
    'of',
    'with',
    'by',
    'from',
    'is',
    'it',
    'as',
    'was',
    'are',
    'be',
    'been',
    'has',
    'have',
    'had',
    'this',
    'that',
    'not',
    'no',
    'do',
    'if',
    'we',
    'you',
    'can',
    'will',
    'what',
    'how',
  };

  /// Raw documents stored at build time so we can rebuild the inverted index
  /// after invalidation without re-reading ObjectBox.
  List<({String chunkId, String text, String packId})> _docs = [];

  /// Per-token inverted index: token → list of (docIdx, term-frequency).
  Map<String, List<({int docIdx, int tf})>> _index = {};

  /// Average document length in tokens. Used by BM25 length normalization.
  double _avgDl = 0;

  /// Whether [buildIndex] has been called and [invalidate] has not been called
  /// since. Checked before search to decide whether to trigger a lazy rebuild.
  bool _built = false;
  bool get isBuilt => _built;

  /// Build the BM25 index from the provided chunks. Safe to call multiple
  /// times — each call fully replaces the previous index.
  void buildIndex(List<({String chunkId, String text, String packId})> chunks) {
    _docs = List.of(chunks, growable: false);

    // Tokenize each document and build the inverted index.
    final docLengths = <int>[];
    final tempIndex = <String, List<({int docIdx, int tf})>>{};

    for (var i = 0; i < _docs.length; i++) {
      final tokens = _tokenize(_docs[i].text);
      docLengths.add(tokens.length);

      // Count term frequencies in this document.
      final tf = <String, int>{};
      for (final t in tokens) {
        tf[t] = (tf[t] ?? 0) + 1;
      }

      for (final entry in tf.entries) {
        tempIndex
            .putIfAbsent(entry.key, () => [])
            .add((docIdx: i, tf: entry.value));
      }
    }

    _index = tempIndex;
    _avgDl = docLengths.isEmpty
        ? 0.0
        : docLengths.reduce((a, b) => a + b) / docLengths.length;
    _built = true;
  }

  /// Mark the index stale. The next search call should trigger a lazy rebuild.
  void invalidate() {
    _built = false;
  }

  /// Search the index for [query], returning up to [topK] results scored by
  /// BM25. If [packIds] is non-empty, only documents matching one of those
  /// pack IDs are considered. Results are sorted by score descending (higher
  /// = better match).
  List<({String chunkId, double score})> search(
    String query, {
    int topK = 10,
    Set<String>? packIds,
  }) {
    if (!_built || _docs.isEmpty) return [];

    final queryTokens = _tokenize(query);
    if (queryTokens.isEmpty) return [];

    final n = _docs.length;
    final scores = <int, double>{};

    // If a lens is set, pre-compute which doc indices are in scope.
    Set<int>? inScope;
    if (packIds != null && packIds.isNotEmpty) {
      final scope = <int>{};
      for (var i = 0; i < _docs.length; i++) {
        if (packIds.contains(_docs[i].packId)) {
          scope.add(i);
        }
      }
      if (scope.isEmpty) return [];
      inScope = scope;
    }

    // Compute doc lengths on demand (cached per search call).
    final docLengths = <int>[];
    for (final doc in _docs) {
      docLengths.add(_tokenize(doc.text).length);
    }

    for (final qt in queryTokens) {
      final postings = _index[qt];
      if (postings == null) continue;

      // IDF: ln((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
      final df = postings.length;
      final idf = log((n - df + 0.5) / (df + 0.5) + 1);

      for (final posting in postings) {
        if (inScope != null && !inScope.contains(posting.docIdx)) continue;

        final dl = docLengths[posting.docIdx];
        final tf = posting.tf;

        // BM25 term score: IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgDl))
        final numerator = tf * (_k1 + 1);
        final denominator = tf + _k1 * (1 - _b + _b * dl / _avgDl);

        scores[posting.docIdx] =
            (scores[posting.docIdx] ?? 0) + idf * numerator / denominator;
      }
    }

    // Sort by score descending, take topK.
    final ranked = scores.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));

    return [
      for (final e in ranked.take(topK))
        (chunkId: _docs[e.key].chunkId, score: e.value),
    ];
  }

  /// Lowercase, strip punctuation, split on whitespace, filter stop words.
  List<String> _tokenize(String text) {
    // Strip punctuation, lowercase, split.
    final cleaned = text.toLowerCase().replaceAll(RegExp(r'[^\w\s]'), ' ');
    final tokens = cleaned.split(RegExp(r'\s+'));
    return [
      for (final t in tokens)
        if (t.isNotEmpty && !_stopWords.contains(t)) t,
    ];
  }
}
