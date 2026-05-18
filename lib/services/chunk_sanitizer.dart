/// One internal "sub-section" of a pack chunk. The source markdown
/// typically uses `####` headings to subdivide a chunk into 3-6 self-
/// contained topics (e.g. inside an "Active Shooter > RUN.HIDE.FIGHT."
/// chunk: RUN, HIDE, FIGHT). Splitting at these boundaries lets retrieval
/// inject the *single most-relevant* sub-section instead of dumping a 4k-
/// char wall and letting Gemma do the picking.
class ChunkSubBlock {
  const ChunkSubBlock({required this.heading, required this.body});
  final String heading;
  final String body;

  /// Markdown rendering of the block. When [heading] is present it's
  /// emitted as `#### Heading` so downstream renderers (and Gemma) see
  /// the same structural cue the source had.
  String get markdown {
    if (heading.isEmpty) return body;
    if (body.isEmpty) return '#### $heading';
    return '#### $heading\n\n$body';
  }
}

/// Split a sanitized chunk into sub-blocks at markdown headings (`##`+).
/// The first sub-block may have an empty [ChunkSubBlock.heading] if the
/// chunk's first non-empty line isn't a heading (some chunks start with a
/// bold lead-in instead). Always returns at least one entry.
List<ChunkSubBlock> splitChunkSubBlocks(String text) {
  final lines = text.split('\n');
  final blocks = <ChunkSubBlock>[];
  var currentHeading = '';
  final currentBody = <String>[];

  void flush() {
    final body = currentBody.join('\n').trim();
    if (currentHeading.isNotEmpty || body.isNotEmpty) {
      blocks.add(ChunkSubBlock(heading: currentHeading, body: body));
    }
    currentHeading = '';
    currentBody.clear();
  }

  final headingPat = RegExp(r'^#{2,6}\s+(.+)$');
  for (final line in lines) {
    final trimmed = line.trim();
    final m = headingPat.firstMatch(trimmed);
    if (m != null) {
      flush();
      currentHeading = m.group(1)!.trim();
    } else {
      currentBody.add(line);
    }
  }
  flush();
  if (blocks.isEmpty) {
    // Fallback: one block with no heading, whole text as body. Keeps the
    // contract that callers always get something to work with.
    return [ChunkSubBlock(heading: '', body: text.trim())];
  }
  return blocks;
}

/// Extract a human-readable title for a chunk — used as the Library card
/// title and the citation chip label so the user sees what's *in* the
/// chunk, not just the HazAdapt taxonomy crumb. Prefers, in order:
///   1. First markdown heading (`#### Foo`)
///   2. Leading bold label like `**EVALUATE:** ...` or `**1. Plan ahead.**`
///   3. First short sentence
/// Returns null if nothing recognizable is found.
String? extractChunkHeading(String text) {
  // Strip leading whitespace, horizontal rules, blank lines.
  final lines = text.split('\n');
  for (final raw in lines) {
    final trimmed = raw.trim();
    if (trimmed.isEmpty || trimmed == '---' || trimmed == '***') continue;

    // 1. Markdown heading.
    final heading = RegExp(r'^#{1,6}\s+(.+)$').firstMatch(trimmed);
    if (heading != null) {
      return _shortenHeading(heading.group(1)!);
    }
    // 2. Bold lead with colon: **EVALUATE:** quickly decide...
    final boldColon = RegExp(r'^\*\*\*?([^*]+?):\*\*\*?').firstMatch(trimmed);
    if (boldColon != null) {
      return _shortenHeading(boldColon.group(1)!);
    }
    // 3. Pure bold/strong line: **Foo bar baz**
    final boldPure = RegExp(r'^\*\*\*?([^*]+?)\*\*\*?\s*$').firstMatch(trimmed);
    if (boldPure != null) {
      return _shortenHeading(boldPure.group(1)!);
    }
    // 4. Fallback: first sentence, capped.
    final firstSentence = trimmed.split(RegExp(r'[.!?]'))[0];
    return _shortenHeading(firstSentence);
  }
  return null;
}

/// Remove the first markdown heading / bold lead-in line from [text] (the
/// one [extractChunkHeading] would surface). Used by the Library reader and
/// the citation detail sheet, where we surface the heading separately as the
/// card title — leaving it inline too would render it twice.
String stripFirstHeading(String text) {
  final lines = text.split('\n');
  final out = <String>[];
  var stripped = false;
  final headingPat = RegExp(r'^#{1,6}\s+');
  final boldColonPat = RegExp(r'^\*\*\*?([^*]+?):\*\*\*?\s*');
  final boldPurePat = RegExp(r'^\*\*\*?([^*]+?)\*\*\*?\s*$');
  for (final raw in lines) {
    if (!stripped) {
      final trimmed = raw.trim();
      if (trimmed.isEmpty || trimmed == '---' || trimmed == '***') continue;
      if (headingPat.hasMatch(trimmed)) {
        stripped = true;
        continue;
      }
      final colon = boldColonPat.firstMatch(trimmed);
      if (colon != null) {
        // Keep whatever text followed "**Label:** " on the same line.
        final remainder = trimmed.replaceFirst(boldColonPat, '').trim();
        if (remainder.isNotEmpty) out.add(remainder);
        stripped = true;
        continue;
      }
      if (boldPurePat.hasMatch(trimmed)) {
        stripped = true;
        continue;
      }
      // First non-empty line wasn't a recognizable heading — keep
      // everything as-is so we don't accidentally strip body text.
      stripped = true;
      out.add(raw);
    } else {
      out.add(raw);
    }
  }
  return out.join('\n').trim();
}

/// Parsed phase ("Preparation" / "Response" / "Recovery") + remaining
/// segments from a raw HazAdapt sectionPath. When the path doesn't lead
/// with a known phase, [phase] is null and [tail] contains everything.
class SectionParts {
  const SectionParts({required this.phase, required this.tail});

  /// One of "Preparation", "Response", "Recovery", or null when the path
  /// doesn't start with a recognized HazAdapt phase.
  final String? phase;

  /// The remaining path segments joined with ` › `. Empty when the path
  /// was just a phase.
  final String tail;
}

/// Decompose a raw sectionPath into a phase + remainder. Lets the Library
/// reader render the phase as a small badge alongside the chunk-extracted
/// title, instead of cramming the whole "Response › For Adults" string
/// into the title slot.
SectionParts parseSectionPath(String? path) {
  if (path == null || path.trim().isEmpty) {
    return const SectionParts(phase: null, tail: '');
  }
  final parts = path
      .split('>')
      .map((s) => s.trim())
      .where((s) => s.isNotEmpty)
      .toList();
  if (parts.isEmpty) return const SectionParts(phase: null, tail: '');

  String? phase;
  final firstLower = parts.first.toLowerCase();
  if (firstLower == 'before (mitigate)' ||
      firstLower == 'before (prepare)' ||
      firstLower == 'before') {
    phase = 'Preparation';
  } else if (firstLower == 'during (react)' || firstLower == 'during') {
    phase = 'Response';
  } else if (firstLower == 'after (recover)' || firstLower == 'after') {
    phase = 'Recovery';
  }

  final rest = phase != null ? parts.skip(1).toList() : parts;
  final tail = rest.map(_renamePhaseSegment).where((s) => s.isNotEmpty).join(' › ');
  return SectionParts(phase: phase, tail: tail);
}

String _shortenHeading(String raw) {
  // Drop trailing punctuation, the [CRITICAL] tag (it's noise in a label),
  // and clamp length so it fits comfortably in a chip / card title.
  var s = raw
      .replaceAll(RegExp(r'\s*\[CRITICAL\]\s*', caseSensitive: false), ' ')
      .replaceAll(RegExp(r'^[\s\*]+|[\s\*:]+$'), '')
      .trim();
  if (s.length > 64) {
    final cut = s.lastIndexOf(' ', 64);
    s = '${s.substring(0, cut > 32 ? cut : 64).trimRight()}…';
  }
  return s;
}

/// English stopwords — kept small on purpose; tiny corpora are sensitive
/// to over-filtering. Used by sub-block relevance scoring so a query like
/// "where do I hide" tokenizes to just {"hide"} when scoring vs sub-block
/// term sets.
const _stopwords = <String>{
  'a', 'an', 'the', 'and', 'or', 'of', 'in', 'to', 'for', 'on', 'at', 'with',
  'is', 'are', 'was', 'were', 'be', 'been', 'being', 'am',
  'it', 'this', 'that', 'these', 'those',
  'i', 'you', 'he', 'she', 'they', 'we', 'me', 'him', 'her', 'them', 'us',
  'my', 'your', 'his', 'their', 'our', 'its',
  'what', 'when', 'where', 'why', 'how', 'who', 'which',
  'do', 'does', 'did', 'done', 'doing',
  'can', 'should', 'would', 'could', 'will', 'may', 'might', 'must',
  'have', 'has', 'had', 'having',
  'as', 'by', 'from', 'if', 'then', 'so', 'but', 'not', 'no', 'yes',
  'too', 'than', 'here', 'there', 'just', 'also', 'about',
};

Set<String> _terms(String s) => s
    .toLowerCase()
    .replaceAll(RegExp(r'[^a-z0-9 ]'), ' ')
    .split(RegExp(r'\s+'))
    .where((w) => w.length > 2 && !_stopwords.contains(w))
    .toSet();

/// Pick the most-relevant slice of [chunkText] for [query], within a
/// [maxChars] budget. The chunk is split at internal markdown headings and
/// each sub-block is scored by term overlap with the query (stopword-
/// filtered set intersection). Top-scoring sub-blocks are concatenated up
/// to the budget. Falls back to a sentence-boundary truncation of the
/// chunk's head when:
///   • the chunk has only one sub-block, or
///   • no sub-block shares any term with the query.
///
/// This is where the "I asked about HIDE but got the whole RUN.HIDE.FIGHT.
/// dump" problem gets fixed: instead of trimming the first 800 chars
/// (which would land on RUN), we score, pick HIDE, and inject that.
String selectRelevantContent({
  required String chunkText,
  required String query,
  required int maxChars,
}) {
  final blocks = splitChunkSubBlocks(chunkText);
  if (blocks.length <= 1) {
    return _truncateAtBoundaryLocal(chunkText, maxChars);
  }
  final qTerms = _terms(query);
  if (qTerms.isEmpty) {
    return _truncateAtBoundaryLocal(chunkText, maxChars);
  }

  // Score each block.
  final scored = <({ChunkSubBlock block, int score, int order})>[];
  for (var i = 0; i < blocks.length; i++) {
    final b = blocks[i];
    final bTerms = _terms('${b.heading} ${b.body}');
    final overlap = qTerms.intersection(bTerms).length;
    scored.add((block: b, score: overlap, order: i));
  }
  // Sort: score desc, then preserve source order on ties.
  scored.sort((a, b) {
    final byScore = b.score.compareTo(a.score);
    return byScore != 0 ? byScore : a.order.compareTo(b.order);
  });

  if (scored.first.score == 0) {
    // No block touched the query — fall back to head-of-chunk truncation.
    // (This is rare since retrieval already cleared the similarity gate,
    // but it can happen when the query and chunk share only stopwords.)
    return _truncateAtBoundaryLocal(chunkText, maxChars);
  }

  // Pack sub-blocks (in score-desc order) until we hit the budget.
  final sb = StringBuffer();
  for (final s in scored) {
    if (s.score == 0) break;
    final piece = s.block.markdown;
    if (sb.isEmpty) {
      sb.write(piece);
    } else if (sb.length + piece.length + 2 <= maxChars) {
      sb
        ..write('\n\n')
        ..write(piece);
    } else {
      break;
    }
  }
  final out = sb.toString();
  return out.length <= maxChars ? out : _truncateAtBoundaryLocal(out, maxChars);
}

/// Local copy of the boundary-aware truncator. Keeps chunk_sanitizer self-
/// contained so callers don't need to import the inference service.
String _truncateAtBoundaryLocal(String text, int limit) {
  if (text.length <= limit) return text;
  final window = text.substring(0, limit);
  final minCut = (limit * 0.7).round();
  final cuts = [
    window.lastIndexOf('\n\n'),
    window.lastIndexOf('. '),
    window.lastIndexOf('! '),
    window.lastIndexOf('? '),
    window.lastIndexOf('\n'),
    window.lastIndexOf(' '),
  ];
  var best = -1;
  for (final c in cuts) {
    if (c >= minCut && c > best) best = c;
  }
  final cutAt = best > 0 ? best : limit;
  return '${text.substring(0, cutAt).trimRight()}…';
}

/// Translate a raw section breadcrumb from the HazAdapt source markdown
/// into a label fit for the citation chip / Library TOC. The source data
/// uses a fixed three-phase taxonomy with parenthetical jargon that means
/// nothing to the user (and looks like a JavaScript framework — see
/// "During (React)"). Strip the jargon and rename the phases.
///
/// Examples:
///   "During (React) > CALL 9-1-1 or your local emergency phone number."
///     → "Response › Call 9-1-1"
///   "Before (Mitigate) > Bug-out bag"
///     → "Preparation › Bug-out bag"
///   "Symptoms (a > b)" → "Symptoms" (other parentheticals stripped too)
String humanizeSectionPath(String? path) {
  if (path == null || path.trim().isEmpty) return '';
  final parts = path
      .split('>')
      .map((s) => _renamePhaseSegment(s.trim()))
      .where((s) => s.isNotEmpty)
      .toList();
  return parts.join(' › ');
}

String _renamePhaseSegment(String s) {
  final lower = s.toLowerCase();
  if (lower == 'before (mitigate)' || lower == 'before') return 'Preparation';
  if (lower == 'during (react)' || lower == 'during') return 'Response';
  if (lower == 'after (recover)' || lower == 'after') return 'Recovery';
  // Drop any other parenthetical clarifications — they're usually
  // categorization metadata that doesn't help the reader.
  var cleaned = s.replaceAll(RegExp(r'\s*\([^)]*\)'), '').trim();
  // Strip markdown emphasis markers (**, *, __, _). Some HazAdapt
  // sectionPaths inherited bold/italic syntax from the source markdown
  // ("**Your goal is to stop the bleeding:**") which rendered literally
  // in the audience badge until we strip these.
  cleaned = cleaned.replaceAll(RegExp(r'[*_]+'), '').trim();
  // Drop trailing colon / period / semicolon — those turn the tail into
  // a sentence fragment that doesn't fit the chip shape.
  cleaned = cleaned.replaceAll(RegExp(r'[:.;]+\s*$'), '').trim();
  // Hide if the cleaned label is too long for a chip. Real audience
  // labels ("For Adults", "For Older Adults") and short topic crumbs
  // ("RUN. HIDE. FIGHT.", "Frostbite") fit under ~40 chars; longer
  // tails are directive sentences that don't help as a badge — better
  // to drop them entirely (caller filters `isNotEmpty`) than to show
  // an ellipsis-clipped sentence-fragment.
  if (cleaned.length > 40) return '';
  // Title-case the first letter so segments read like proper labels.
  if (cleaned.isNotEmpty) {
    cleaned = cleaned[0].toUpperCase() + cleaned.substring(1);
  }
  return cleaned;
}

/// Cleans markdown chunk text before injection into the LLM prompt.
///
/// The bundled packs were embedded long before we noticed Gemma 4 sometimes
/// echoes weird artifacts straight back into its replies — zero-width chars,
/// smart quotes that decode into multi-token sequences, stray HTML, footnote
/// markers. The model is small enough that it mirrors what it sees in context,
/// so dirty input → dirty output (this was the "degenerated tokens" symptom).
///
/// Sanitizing at injection time fixes the symptom without forcing a re-embed
/// of every pack: the *embedding* stays whatever it was when the JSON was
/// generated; only the *text we hand back to Gemma* is cleaned.
String sanitizeChunkText(String input) {
  // 1. Walk the string and drop invisible / control codepoints. Doing this
  //    explicitly (instead of with a regex character class containing literal
  //    high-codepoint chars) keeps the source file safe to round-trip through
  //    editors that might silently re-normalize escapes.
  final buf = StringBuffer();
  for (final cu in input.runes) {
    if (_shouldDrop(cu)) continue;
    buf.writeCharCode(cu);
  }
  var s = buf.toString();

  // 2. Fold smart quotes / dashes / ellipsis to plain ASCII. Gemma sometimes
  //    splits the multi-byte form mid-token and then drifts into pretraining
  //    noise. ASCII forms are single-token and stable.
  s = s
      .replaceAll('“', '"') // “ left double quote
      .replaceAll('”', '"') // ” right double quote
      .replaceAll('‘', "'") // ‘ left single quote
      .replaceAll('’', "'") // ’ right single quote / apostrophe
      .replaceAll('—', '-') // — em dash
      .replaceAll('–', '-') // – en dash
      .replaceAll('…', '...'); // … horizontal ellipsis

  // 3. Strip HTML tags but keep their text content. The bundled markdown
  //    occasionally has inline <sub>, <sup>, <br> that survived conversion.
  s = s.replaceAll(RegExp(r'<[a-zA-Z/][^>]*>'), '');

  // 4. Drop Pandoc-style footnote markers like [^1] and citation placeholders
  //    like {cite:foo}. These survive markdown conversion but mean nothing to
  //    the model — it just learns to emit similar gibberish back.
  s = s
      .replaceAll(RegExp(r'\[\^[^\]]*\]'), '')
      .replaceAll(RegExp(r'\{cite:[^}]*\}'), '');

  // 5. Collapse runs of 3+ newlines to a single blank line. Keeps paragraph
  //    breaks but flattens the ragged spacing some sources have.
  s = s.replaceAll(RegExp(r'\n{3,}'), '\n\n');

  // 6. Collapse intra-line runs of spaces/tabs.
  s = s.replaceAll(RegExp(r'[ \t]+'), ' ');

  // 7. Strip trailing whitespace per line, then trim ends.
  s = s.split('\n').map((l) => l.trimRight()).join('\n');

  // 8. Collapse consecutive identical lines. Many source chunks open with
  //    their heading repeated verbatim ("Avalanche Starting Below You\n\n
  //    Avalanche Starting Below You\n\n...") — once as a markdown heading,
  //    once as a paragraph rephrase. The duplication trains Gemma to echo
  //    in its own replies.
  final lines = s.split('\n');
  final out = <String>[];
  for (final l in lines) {
    final norm = l.trim();
    final lastNorm = out.isEmpty ? null : out.last.trim();
    // Blank lines pass through (we need them for paragraph breaks); only
    // dedupe non-empty consecutive duplicates.
    if (norm.isEmpty || lastNorm == null || norm != lastNorm) {
      out.add(l);
    }
  }
  s = out.join('\n');

  return s.trim();
}

bool _shouldDrop(int cu) {
  // ASCII control chars except \n (0x0A) and \t (0x09).
  if (cu < 0x20) return cu != 0x09 && cu != 0x0A;
  if (cu == 0x7F) return true;
  // Zero-width space, ZWNJ, ZWJ, LRM, RLM (0x200B–0x200F).
  if (cu >= 0x200B && cu <= 0x200F) return true;
  // Line separator, paragraph separator (0x2028, 0x2029).
  if (cu == 0x2028 || cu == 0x2029) return true;
  // Bidi formatting: LRE, RLE, PDF, LRO, RLO (0x202A–0x202E).
  if (cu >= 0x202A && cu <= 0x202E) return true;
  // Word joiner (0x2060) through invisible operators (0x2064).
  if (cu >= 0x2060 && cu <= 0x2064) return true;
  // BOM / zero-width no-break space.
  if (cu == 0xFEFF) return true;
  return false;
}
