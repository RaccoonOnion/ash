/// Lightweight fuzzy matcher for in-app search (chat titles, pack names).
///
/// Returns a score where higher is better, or `null` if `query` has no match
/// in `text`. Scoring blends:
///  * substring hit (large bonus, extra if it lands on a word boundary)
///  * subsequence match — every query character must appear in order
///  * adjacency bonus — consecutive matches score more than gaps
///
/// All comparisons are case-insensitive and unicode is left as-is (good
/// enough for English; we never lowercase characters that don't change).
int? fuzzyScore(String query, String text) {
  if (query.isEmpty) return 0;
  final q = query.toLowerCase().trim();
  final t = text.toLowerCase();
  if (q.isEmpty) return 0;

  // Fast path: exact substring is the best possible match. Word-boundary
  // hits beat mid-word hits ("first aid" should rank above "firstly").
  final substr = t.indexOf(q);
  if (substr >= 0) {
    final atStart = substr == 0;
    final wordBoundary = atStart || _isWordBoundary(t.codeUnitAt(substr - 1));
    return 1000 + (atStart ? 200 : 0) + (wordBoundary ? 100 : 0) - substr;
  }

  // Subsequence match: walk both strings in order; every query char must
  // appear in `text` in the right sequence.
  var ti = 0;
  var score = 0;
  var prevMatch = -2;
  for (var qi = 0; qi < q.length; qi++) {
    final qc = q.codeUnitAt(qi);
    while (ti < t.length && t.codeUnitAt(ti) != qc) {
      ti++;
    }
    if (ti >= t.length) return null;
    // Adjacency bonus: characters that match back-to-back score double.
    if (ti == prevMatch + 1) score += 4;
    score += 1;
    // Word-start bonus.
    if (ti == 0 || _isWordBoundary(t.codeUnitAt(ti - 1))) score += 2;
    prevMatch = ti;
    ti++;
  }
  return score;
}

/// True if the given code unit is a separator that would start a new "word"
/// (whitespace, punctuation, common dashes/slashes).
bool _isWordBoundary(int codeUnit) {
  // ' ' \t \n - _ . , : ; / \ ( ) [ ]
  return codeUnit == 0x20 ||
      codeUnit == 0x09 ||
      codeUnit == 0x0A ||
      codeUnit == 0x2D ||
      codeUnit == 0x5F ||
      codeUnit == 0x2E ||
      codeUnit == 0x2C ||
      codeUnit == 0x3A ||
      codeUnit == 0x3B ||
      codeUnit == 0x2F ||
      codeUnit == 0x5C ||
      codeUnit == 0x28 ||
      codeUnit == 0x29 ||
      codeUnit == 0x5B ||
      codeUnit == 0x5D;
}

/// Score a record across multiple weighted fields. Returns the max score
/// found across fields (so a title hit dominates a preview hit), or `null`
/// if nothing matches.
int? fuzzyScoreFields(String query, List<({String text, double weight})> fields) {
  int? best;
  for (final f in fields) {
    final s = fuzzyScore(query, f.text);
    if (s == null) continue;
    final weighted = (s * f.weight).round();
    if (best == null || weighted > best) best = weighted;
  }
  return best;
}
