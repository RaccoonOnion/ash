import 'package:flutter/material.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';

import '../models/pack.dart';
import '../services/chunk_sanitizer.dart';
import '../services/inference_service.dart';
import '../theme/ash_theme.dart';
import '../utils/fuzzy.dart';
import '../widgets/buttons.dart';
import '../widgets/glass.dart';
import '../widgets/pack_art.dart';
import '../widgets/text_field.dart';

/// Library reader — browse a pack as Q&A propositions. Each chunk is one
/// `(question, answer)` proposition; the question lives at the top of the
/// chunk's text as a `####` heading. The reader extracts that heading as
/// the card title (Socratic), groups cards by lifecycle phase
/// (Preparation / Response / Recovery), and offers a search bar that
/// doubles as an Ask-Ash launcher when the user's typed query doesn't
/// fuzzy-match anything in the pack closely.
///
/// Three "Ask Ash" entry points coexist:
///   • Per-card "Ask Ash about this"
///     → opens chat with this question pre-filled, lens scoped to the pack.
///   • Search-bar "Ask Ash: '...'" CTA when matches are weak
///     → opens chat with the typed query pre-filled, lens scoped to the pack.
///   • Floating "Ask Ash about {pack}"
///     → opens chat scoped to the pack with no pre-filled message.
class PackReaderScreen extends StatefulWidget {
  const PackReaderScreen({
    required this.pack,
    required this.service,
    required this.onAskAboutPack,
    required this.onAskAboutSection,
    this.focusChunkId,
    super.key,
  });

  final Pack pack;
  final InferenceService service;

  /// Floating CTA — chat scoped to the pack, no seed message.
  final VoidCallback onAskAboutPack;

  /// Per-card CTA + search-bar CTA — chat scoped to the pack with the given
  /// string seeded as the user's first message. The pack reader calls this
  /// with the proposition's question (per-card) or the user's typed query
  /// (search-bar fallback).
  final void Function(String seedMessage) onAskAboutSection;

  /// When non-null, the reader auto-scrolls to the card whose
  /// [LibraryChunk.chunkId] matches and briefly pulses its border so the
  /// user knows exactly which proposition was cited. Used by citation
  /// chip deep-links.
  final String? focusChunkId;

  @override
  State<PackReaderScreen> createState() => _PackReaderScreenState();
}

class _PackReaderScreenState extends State<PackReaderScreen> {
  late Future<List<LibraryChunk>> _chunksFuture;
  final _searchController = TextEditingController();
  String _query = '';

  /// Fuzzy score threshold above which a card is considered a "good"
  /// match for the user's query. Below this, we surface the "Ask Ash this
  /// question" CTA at the top of the result list.
  static const _strongMatchScore = 600;

  /// Key per rendered card, keyed by chunkId, used to drive
  /// [Scrollable.ensureVisible] when arriving via a citation deep-link.
  /// Re-allocated on every chunks-future load (rare — once per screen).
  final Map<String, GlobalKey> _cardKeys = {};

  /// True until the deep-link scroll-and-pulse has fired. The pulse uses
  /// this to draw the accent border on the targeted card for ~1.6s on
  /// first visibility.
  bool _deepLinkPulse = false;
  bool _deepLinkScrollScheduled = false;

  @override
  void initState() {
    super.initState();
    _chunksFuture = widget.service.readPack(widget.pack.id);
    if (widget.focusChunkId != null) {
      _deepLinkPulse = true;
    }
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  /// Looks up (or lazily allocates) the [GlobalKey] for a given chunk's
  /// card so the proposition can be located by chunkId at scroll-time.
  GlobalKey _keyFor(String chunkId) =>
      _cardKeys.putIfAbsent(chunkId, () => GlobalKey());

  /// Schedule a single post-frame scroll to the focus chunk. The reader
  /// is built lazily inside FutureBuilder, so this runs after the first
  /// frame where chunks are available. Pulse-fade kicks in alongside the
  /// scroll and clears itself after ~1.6s.
  void _maybeScheduleDeepLinkScroll(List<LibraryChunk> chunks) {
    if (_deepLinkScrollScheduled) return;
    final target = widget.focusChunkId;
    if (target == null) return;
    if (!chunks.any((c) => c.chunkId == target)) return;
    _deepLinkScrollScheduled = true;
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final ctx = _cardKeys[target]?.currentContext;
      if (ctx == null) return;
      Scrollable.ensureVisible(
        ctx,
        duration: const Duration(milliseconds: 420),
        curve: Curves.easeOutCubic,
        alignment: 0.1,
      );
      Future.delayed(const Duration(milliseconds: 1600), () {
        if (mounted) setState(() => _deepLinkPulse = false);
      });
    });
  }

  /// Group chunks by lifecycle phase. The proposition pipeline tags every
  /// chunk's sectionPath with a "Before (Prepare) / During (React) / After
  /// (Recover)" prefix; [parseSectionPath] maps those to human labels.
  Map<String, List<LibraryChunk>> _groupByPhase(List<LibraryChunk> chunks) {
    final byPhase = <String, List<LibraryChunk>>{
      'Preparation': [],
      'Response': [],
      'Recovery': [],
      'Other': [],
    };
    for (final ch in chunks) {
      final phase = parseSectionPath(ch.sectionPath).phase ?? 'Other';
      byPhase.putIfAbsent(phase, () => []).add(ch);
    }
    return byPhase;
  }

  List<({LibraryChunk chunk, int score})> _scoreByQuery(
    List<LibraryChunk> chunks,
    String query,
  ) {
    final q = query.trim();
    if (q.isEmpty) return const [];
    final out = <({LibraryChunk chunk, int score})>[];
    for (final c in chunks) {
      // Score against the chunk's question (first #### heading) primarily,
      // and the body weakly so a question term that lives in the answer
      // still surfaces.
      final question = extractChunkHeading(c.text) ?? '';
      final s = fuzzyScoreFields(q, [
        (text: question, weight: 1.0),
        (text: c.text, weight: 0.25),
      ]);
      if (s == null) continue;
      out.add((chunk: c, score: s));
    }
    out.sort((a, b) => b.score.compareTo(a.score));
    return out;
  }

  @override
  Widget build(BuildContext context) {
    final c = ash(context);
    return Scaffold(
      backgroundColor: AshColors.dark.bg,
      body: SafeArea(
        child: Column(
          children: [
            // Header — pack identity.
            Padding(
              padding: const EdgeInsets.fromLTRB(8, 8, 16, 8),
              child: Row(
                children: [
                  IconBtn(
                    icon: Icons.arrow_back_ios_new,
                    onTap: () => Navigator.of(context).pop(),
                  ),
                  const SizedBox(width: 8),
                  PackArt(
                    size: 36,
                    radius: 10,
                    fromColor: widget.pack.artFrom,
                    toColor: widget.pack.artTo,
                    glyph: widget.pack.glyph,
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text(
                          widget.pack.name,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: TextStyle(
                            color: c.text,
                            fontSize: 16,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                        Text(
                          '${widget.pack.entries} entries · '
                          'v${widget.pack.version} · '
                          'by ${widget.pack.author}',
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: TextStyle(color: c.textDim, fontSize: 11),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),

            // Search bar — filters propositions; doubles as an Ask-Ash
            // launcher when nothing in the pack matches the query well.
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 4, 20, 8),
              child: AshTextField(
                hint: 'Search this pack or ask a new question…',
                controller: _searchController,
                leadingIcon: Icons.search,
                onChanged: (v) => setState(() => _query = v),
              ),
            ),

            Expanded(
              child: FutureBuilder<List<LibraryChunk>>(
                future: _chunksFuture,
                builder: (context, snap) {
                  if (snap.connectionState != ConnectionState.done) {
                    return const Center(child: CircularProgressIndicator());
                  }
                  if (snap.hasError) {
                    return Center(
                      child: Text(
                        'Failed to read pack: ${snap.error}',
                        style: TextStyle(color: c.textDim, fontSize: 13),
                      ),
                    );
                  }
                  final chunks = snap.data ?? const [];
                  if (chunks.isEmpty) {
                    return Center(
                      child: Text(
                        'This pack has no readable entries yet.',
                        style: TextStyle(color: c.textDim, fontSize: 13),
                      ),
                    );
                  }
                  // Schedule the deep-link scroll once chunks land. Safe
                  // to call from inside build — it's idempotent and only
                  // schedules a post-frame callback the first time.
                  _maybeScheduleDeepLinkScroll(chunks);
                  final hasQuery = _query.trim().isNotEmpty;
                  return hasQuery
                      ? _buildSearchResults(c, chunks)
                      : _buildPhaseGrouped(c, chunks);
                },
              ),
            ),
          ],
        ),
      ),
      // Bottom CTA: ask Ash about the pack as a whole, no seed message.
      floatingActionButton: Padding(
        padding: const EdgeInsets.only(bottom: 16),
        child: FloatingActionButton.extended(
          backgroundColor: c.accent,
          onPressed: widget.onAskAboutPack,
          icon: const Icon(Icons.chat_bubble_outline, color: Colors.white),
          label: Text(
            'Ask Ash about ${widget.pack.name}',
            style: const TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }

  /// Default view — fixed phase order (Preparation → Response → Recovery)
  /// so the survival flow reads top-to-bottom.
  ///
  /// Uses [SingleChildScrollView]+[Column] rather than [ListView] so every
  /// card is laid out eagerly. The citation deep-link relies on
  /// [Scrollable.ensureVisible], which silently no-ops on items whose
  /// RenderObjects haven't been inflated yet — a lazy SliverList wouldn't
  /// build off-screen cards in time for the post-frame scroll, so the
  /// reader would just sit at the top. Pack sizes (≤50 cards) make eager
  /// layout an easy trade.
  Widget _buildPhaseGrouped(AshColors c, List<LibraryChunk> chunks) {
    final byPhase = _groupByPhase(chunks);
    const phaseOrder = ['Preparation', 'Response', 'Recovery', 'Other'];
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(20, 4, 20, 120),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          for (final phase in phaseOrder)
            if ((byPhase[phase] ?? const []).isNotEmpty) ...[
              _PhaseHeader(
                c: c,
                label: phase,
                count: byPhase[phase]!.length,
              ),
              const SizedBox(height: 10),
              for (final chunk in byPhase[phase]!) ...[
                _PropositionCard(
                  c: c,
                  chunk: chunk,
                  cardKey: _keyFor(chunk.chunkId),
                  highlight: _deepLinkPulse &&
                      widget.focusChunkId == chunk.chunkId,
                  onAsk: () => widget.onAskAboutSection(
                      extractChunkHeading(chunk.text) ?? widget.pack.name),
                ),
                const SizedBox(height: 10),
              ],
              const SizedBox(height: 14),
            ],
        ],
      ),
    );
  }

  /// Search-mode view — rank by fuzzy score against the query. When the
  /// best score is below [_strongMatchScore], surface the "Ask Ash this
  /// question" CTA at the top so the user has an obvious next move when
  /// the pack doesn't cover their exact question.
  Widget _buildSearchResults(AshColors c, List<LibraryChunk> chunks) {
    final query = _query.trim();
    final scored = _scoreByQuery(chunks, query);
    final hasStrongMatch =
        scored.isNotEmpty && scored.first.score >= _strongMatchScore;
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(20, 4, 20, 120),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          if (!hasStrongMatch)
            _AskAshCta(
              c: c,
              query: query,
              onTap: () => widget.onAskAboutSection(query),
            ),
          if (!hasStrongMatch && scored.isNotEmpty)
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 12),
              child: Text(
                'Loose matches in this pack:',
                style: TextStyle(
                  color: c.textDim,
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                  letterSpacing: 0.6,
                ),
              ),
            ),
          if (scored.isEmpty)
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 24),
              child: Text(
                'Nothing in this pack matches "$query".',
                textAlign: TextAlign.center,
                style: TextStyle(color: c.textDim, fontSize: 13),
              ),
            )
          else
            for (final s in scored) ...[
              _PropositionCard(
                c: c,
                chunk: s.chunk,
                cardKey: _keyFor(s.chunk.chunkId),
                highlight: _deepLinkPulse &&
                    widget.focusChunkId == s.chunk.chunkId,
                onAsk: () => widget.onAskAboutSection(
                  extractChunkHeading(s.chunk.text) ?? widget.pack.name,
                ),
              ),
              const SizedBox(height: 10),
            ],
        ],
      ),
    );
  }
}

/// Single Q&A proposition. The chunk's text is `#### Question\n\nAnswer`,
/// so we lift the question via [extractChunkHeading] for the title and
/// render the answer body via [stripFirstHeading] so the question doesn't
/// appear twice. Audience badge (Adults / Kids / Pets / etc.) renders when
/// the sectionPath carries one.
class _PropositionCard extends StatelessWidget {
  const _PropositionCard({
    required this.c,
    required this.chunk,
    required this.onAsk,
    this.cardKey,
    this.highlight = false,
  });

  final AshColors c;
  final LibraryChunk chunk;
  final VoidCallback onAsk;

  /// GlobalKey attached to the outer card so the parent can call
  /// [Scrollable.ensureVisible] on it for deep-link landings.
  final Key? cardKey;

  /// When true, draw an accent ring + glow around the card to make the
  /// deep-link target unmistakable. Fades back to the default Glass look
  /// after the pulse window in the parent.
  final bool highlight;

  @override
  Widget build(BuildContext context) {
    final parts = parseSectionPath(chunk.sectionPath);
    final question = extractChunkHeading(chunk.text);
    final title = question?.isNotEmpty == true ? question! : 'Section';
    final body =
        question != null ? stripFirstHeading(chunk.text) : chunk.text;

    final card = Glass(
      radius: 18,
      padding: const EdgeInsets.all(14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Audience badge if present. Phase is encoded in the section
          // header above the card group, so no phase badge here.
          if (parts.tail.isNotEmpty) ...[
            _AudienceBadge(c: c, label: parts.tail),
            const SizedBox(height: 8),
          ],
          // Question — the card's title.
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                width: 6,
                height: 18,
                margin: const EdgeInsets.only(top: 2),
                decoration: BoxDecoration(
                  color: c.accent,
                  borderRadius: BorderRadius.circular(3),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  title,
                  style: TextStyle(
                    color: c.text,
                    fontSize: 15,
                    fontWeight: FontWeight.w700,
                    height: 1.3,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          MarkdownBody(
            data: body,
            selectable: true,
            softLineBreak: true,
            styleSheet: _readerStyle(c),
          ),
          const SizedBox(height: 12),
          GestureDetector(
            onTap: onAsk,
            behavior: HitTestBehavior.opaque,
            child: Container(
              padding:
                  const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              decoration: BoxDecoration(
                color: c.accentSoft,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.psychology_outlined, size: 14, color: c.accent),
                  const SizedBox(width: 6),
                  Text(
                    'Ask Ash about this',
                    style: TextStyle(
                      color: c.accent,
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );

    // Deep-link landings get a brief accent ring + glow so the user
    // sees exactly which proposition was cited. We animate the wrapping
    // container so the highlight fades out smoothly when the pulse
    // window in the parent expires.
    return KeyedSubtree(
      key: cardKey,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 480),
        curve: Curves.easeOut,
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: highlight ? c.accent : Colors.transparent,
            width: highlight ? 1.5 : 0,
          ),
          boxShadow: highlight
              ? [BoxShadow(color: c.accentGlow, blurRadius: 20)]
              : const [],
        ),
        child: card,
      ),
    );
  }

  MarkdownStyleSheet _readerStyle(AshColors c) {
    final base = TextStyle(color: c.text, fontSize: 14, height: 1.5);
    return MarkdownStyleSheet(
      p: base,
      h1: base.copyWith(fontSize: 18, fontWeight: FontWeight.w700, height: 1.3),
      h2: base.copyWith(fontSize: 16, fontWeight: FontWeight.w700, height: 1.3),
      h3: base.copyWith(fontSize: 15, fontWeight: FontWeight.w700, height: 1.3),
      h4: base.copyWith(fontSize: 14, fontWeight: FontWeight.w700),
      strong: base.copyWith(fontWeight: FontWeight.w700),
      em: base.copyWith(fontStyle: FontStyle.italic),
      listBullet: base.copyWith(color: c.accent),
      blockquoteDecoration: BoxDecoration(
        color: c.surfaceElev.withValues(alpha: 0.4),
        border: Border(left: BorderSide(color: c.accent, width: 3)),
      ),
      blockquotePadding: const EdgeInsets.fromLTRB(10, 6, 8, 6),
    );
  }
}

/// Top-of-phase heading rendered above each phase group (Preparation,
/// Response, Recovery). Sets up the survival-flow reading order.
class _PhaseHeader extends StatelessWidget {
  const _PhaseHeader({
    required this.c,
    required this.label,
    required this.count,
  });

  final AshColors c;
  final String label;
  final int count;

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Text(
          label.toUpperCase(),
          style: TextStyle(
            color: c.accent,
            fontSize: 11,
            fontWeight: FontWeight.w700,
            letterSpacing: 0.8,
          ),
        ),
        const SizedBox(width: 6),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 1),
          decoration: BoxDecoration(
            color: c.surfaceElev,
            borderRadius: BorderRadius.circular(8),
            border: Border.all(color: c.border, width: 0.5),
          ),
          child: Text(
            '$count',
            style: TextStyle(
              color: c.textMuted,
              fontSize: 10,
              fontWeight: FontWeight.w700,
            ),
          ),
        ),
      ],
    );
  }
}

/// Top-of-results CTA shown when the search query doesn't fuzzy-match any
/// proposition strongly. Pivots the user from "browse" mode into "ask"
/// mode without making them retype their query.
class _AskAshCta extends StatelessWidget {
  const _AskAshCta({
    required this.c,
    required this.query,
    required this.onTap,
  });

  final AshColors c;
  final String query;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      behavior: HitTestBehavior.opaque,
      child: Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: c.accentSoft,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: c.accent, width: 1),
          boxShadow: [
            BoxShadow(color: c.accentGlow, blurRadius: 16),
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
              ),
              alignment: Alignment.center,
              child: const Icon(
                Icons.auto_awesome,
                size: 18,
                color: Colors.white,
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    'Ask Ash this question',
                    style: TextStyle(
                      color: c.accent,
                      fontSize: 13,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    '"$query"',
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(
                      color: c.text,
                      fontSize: 13,
                      fontStyle: FontStyle.italic,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(width: 8),
            Icon(Icons.arrow_forward_ios, color: c.accent, size: 14),
          ],
        ),
      ),
    );
  }
}

/// Audience chip — replaces the phase-badge from earlier rounds; phase is
/// now communicated by the surrounding section header.
class _AudienceBadge extends StatelessWidget {
  const _AudienceBadge({required this.c, required this.label});
  final AshColors c;
  final String label;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: c.surfaceElev,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: c.border, width: 0.5),
      ),
      child: Text(
        label,
        style: TextStyle(
          color: c.textMuted,
          fontSize: 10,
          fontWeight: FontWeight.w600,
          letterSpacing: 0.3,
        ),
      ),
    );
  }
}
