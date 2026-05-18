import 'dart:async';
import 'package:flutter/material.dart' hide Chip;
import '../theme/ash_theme.dart';
import '../models/pack.dart';
import '../services/inference_service.dart';
import '../utils/fuzzy.dart';
import '../widgets/glass.dart';
import '../widgets/text_field.dart';
import '../widgets/pack_art.dart';

/// Filter mode for the Knowledge screen — either show every pack (default)
/// or just the ones currently added to the assistant's search index.
enum _PackFilter { all, added }

/// Knowledge management screen. Single combined list (no Library/Store
/// tab split — both tabs were showing the same underlying inventory in
/// different states; merging makes the "add" / "remove" actions and the
/// pack state directly visible together).
///
/// Layout:
///   • Search bar (fuzzy + semantic-recommend overlay when typing)
///   • Filter chip: All packs / Added only
///   • Packs grouped by tier (Essential / Standard / Comprehensive)
///   • Each card shows pack info and an Add or Remove action
///   • Tap an *added* card → opens the Library reader for that pack
class KnowledgeScreen extends StatefulWidget {
  const KnowledgeScreen({
    required this.packs,
    required this.onPacksChanged,
    required this.onAddPack,
    required this.onRemovePack,
    this.onOpenPack,
    this.onRankPacks,
    super.key,
  });

  final List<Pack> packs;
  final ValueChanged<List<Pack>> onPacksChanged;

  /// Add the pack to the on-device search index. Receives a progress
  /// callback so we can show real download progress for remote packs (or
  /// pulse a quick 5%→100% step for bundled ones, which finish in ms).
  final Future<void> Function(Pack pack, void Function(double)? onProgress)
      onAddPack;

  /// Remove the pack's chunks from the on-device search index. The
  /// underlying bundled JSON stays in the app binary, so Re-adding is a
  /// quick re-import.
  final Future<void> Function(String packId) onRemovePack;

  /// Open the Library reader for a pack. Only meaningful for already-added
  /// packs (the reader sources from the on-device search index).
  final ValueChanged<String>? onOpenPack;

  /// Semantic ranking using the same MiniLM embeddings the runtime uses
  /// for retrieval. Drives the "Recommended for you" overlay above the
  /// tier list when the user types into the search bar.
  final Future<List<PackRanking>> Function(String query)? onRankPacks;

  @override
  State<KnowledgeScreen> createState() => _KnowledgeScreenState();
}

class _KnowledgeScreenState extends State<KnowledgeScreen> {
  final _queryController = TextEditingController();
  String _query = '';
  _PackFilter _filter = _PackFilter.all;

  /// packId → add progress (0..1). Driven by the real import work.
  final Map<String, double> _progress = {};

  /// Top-of-list semantic recommendations for the current query. Debounced.
  List<PackRanking> _recs = const [];
  Timer? _recsDebounce;
  String _recsForQuery = '';

  /// Threshold below which a pack isn't "recommended". MiniLM cosine
  /// similarity for related text in this corpus runs ~0.2–0.5; 0.32
  /// keeps the bar high enough that empty/off-topic queries don't
  /// surface random packs.
  static const _recScoreThreshold = 0.32;

  @override
  void dispose() {
    _queryController.dispose();
    _recsDebounce?.cancel();
    super.dispose();
  }

  /// Fuzzy-score the input pack list against [_query]. Returns the original
  /// order when query is empty.
  List<Pack> _fuzzyFilter(List<Pack> packs, String query) {
    final q = query.trim();
    if (q.isEmpty) return packs;
    final scored = <({Pack pack, int score})>[];
    for (final p in packs) {
      final s = fuzzyScoreFields(q, [
        (text: p.name, weight: 1.0),
        (text: p.summary, weight: 0.5),
        (text: p.author, weight: 0.3),
      ]);
      if (s == null) continue;
      scored.add((pack: p, score: s));
    }
    scored.sort((a, b) => b.score.compareTo(a.score));
    return scored.map((e) => e.pack).toList();
  }

  /// Kick off a (debounced) semantic ranking pass. Embedding the query is
  /// fast (a few ms via MiniLM) but firing on every keystroke is wasteful.
  void _scheduleRankRefresh() {
    _recsDebounce?.cancel();
    final ranker = widget.onRankPacks;
    if (ranker == null) {
      if (_recs.isNotEmpty) setState(() => _recs = const []);
      return;
    }
    final q = _query.trim();
    if (q.length < 2) {
      if (_recs.isNotEmpty || _recsForQuery.isNotEmpty) {
        setState(() {
          _recs = const [];
          _recsForQuery = '';
        });
      }
      return;
    }
    _recsDebounce = Timer(const Duration(milliseconds: 180), () async {
      try {
        final out = await ranker(q);
        if (!mounted) return;
        if (q != _query.trim()) return; // stale result
        final filtered = out
            .where((r) => r.score >= _recScoreThreshold)
            .take(3)
            .toList();
        setState(() {
          _recs = filtered;
          _recsForQuery = q;
        });
      } catch (_) {
        if (mounted) setState(() => _recs = const []);
      }
    });
  }

  List<Pack> get _added => widget.packs.where((p) => p.installed).toList();

  double get _totalBytes => _added.fold(0.0, (sum, p) => sum + p.bytes);

  Future<void> _addPack(Pack pack) async {
    if (_progress.containsKey(pack.id)) return;
    setState(() => _progress[pack.id] = 0.05);
    try {
      await widget.onAddPack(pack, (p) {
        if (!mounted) return;
        setState(() => _progress[pack.id] =
            p < 0.05 ? 0.05 : p.clamp(0.0, 1.0));
      });
    } catch (_) {
      if (mounted) setState(() => _progress.remove(pack.id));
      return;
    }
    if (!mounted) return;
    setState(() => _progress[pack.id] = 1.0);
    final newPacks = widget.packs.map((p) {
      if (p.id == pack.id) return p.copyWith(installed: true);
      return p;
    }).toList();
    widget.onPacksChanged(newPacks);
    Future.delayed(const Duration(milliseconds: 300), () {
      if (mounted) setState(() => _progress.remove(pack.id));
    });
  }

  Future<void> _removePack(Pack pack) async {
    await widget.onRemovePack(pack.id);
    final newPacks = widget.packs.map((p) {
      if (p.id == pack.id) return p.copyWith(installed: false);
      return p;
    }).toList();
    widget.onPacksChanged(newPacks);
  }

  @override
  Widget build(BuildContext context) {
    final c = ash(context);
    final hasQuery = _query.trim().isNotEmpty;

    // Apply the filter chip first, then the fuzzy filter from the search
    // box. Recommendations are derived from `widget.packs` directly so
    // they can surface a pack the user has hidden via the "Added only"
    // chip when relevant — we just label them "recommended".
    final byFilter = _filter == _PackFilter.added ? _added : widget.packs;
    final filtered = _fuzzyFilter(byFilter, _query);

    // Tier-group the filtered set.
    final byTier = <PackTier, List<Pack>>{
      PackTier.essential: [],
      PackTier.standard: [],
      PackTier.comprehensive: [],
    };
    for (final p in filtered) {
      byTier[p.tier]!.add(p);
    }

    return SafeArea(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 16, 20, 0),
            child: Row(
              children: [
                Text(
                  'Knowledge',
                  style: TextStyle(
                    color: c.text,
                    fontSize: 36,
                    fontWeight: FontWeight.w400,
                  ),
                ),
                const SizedBox(width: 12),
                _StorageChip(c: c, bytes: _totalBytes),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 6, 20, 0),
            child: Text(
              'Add a pack to make it searchable by Ash. Tap an added pack to read it.',
              style: TextStyle(color: c.textMuted, fontSize: 14),
            ),
          ),

          // Search bar
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 14, 20, 0),
            child: AshTextField(
              hint: 'Search packs or describe a situation…',
              controller: _queryController,
              leadingIcon: Icons.search,
              onChanged: (v) {
                setState(() => _query = v);
                _scheduleRankRefresh();
              },
            ),
          ),

          // Filter chips
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 10, 20, 0),
            child: Row(
              children: [
                _FilterChip(
                  c: c,
                  label: 'All packs',
                  count: widget.packs.length,
                  selected: _filter == _PackFilter.all,
                  onTap: () => setState(() => _filter = _PackFilter.all),
                ),
                const SizedBox(width: 8),
                _FilterChip(
                  c: c,
                  label: 'Added',
                  count: _added.length,
                  selected: _filter == _PackFilter.added,
                  onTap: () => setState(() => _filter = _PackFilter.added),
                ),
              ],
            ),
          ),

          // List
          Expanded(
            child: ListView(
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 40),
              children: [
                // Semantic recommendations row (only when query is active
                // AND we have packs above the similarity threshold).
                if (_recs.isNotEmpty &&
                    hasQuery &&
                    _recsForQuery == _query.trim()) ...[
                  _RecommendedHeader(c: c),
                  const SizedBox(height: 8),
                  for (final r in _recs) ...[
                    _buildRecommendedCard(c, r),
                    const SizedBox(height: 10),
                  ],
                  const SizedBox(height: 14),
                ],

                if (filtered.isEmpty)
                  Padding(
                    padding: const EdgeInsets.symmetric(vertical: 32),
                    child: Center(
                      child: Text(
                        hasQuery
                            ? 'No packs match "$_query".'
                            : (_filter == _PackFilter.added
                                ? "You haven't added any packs yet."
                                : 'No packs available.'),
                        textAlign: TextAlign.center,
                        style: TextStyle(color: c.textDim, fontSize: 13),
                      ),
                    ),
                  )
                else
                  for (final tier in PackTier.values)
                    if (byTier[tier]!.isNotEmpty) ...[
                      _TierHeader(
                        c: c,
                        tier: tier,
                        count: byTier[tier]!.length,
                        showBlurb: !hasQuery,
                      ),
                      const SizedBox(height: 8),
                      for (final pack in byTier[tier]!) ...[
                        _buildPackCard(c, pack),
                        const SizedBox(height: 10),
                      ],
                      const SizedBox(height: 14),
                    ],
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildRecommendedCard(AshColors c, PackRanking r) {
    final pack = widget.packs.where((p) => p.id == r.packId).firstOrNull;
    if (pack == null) return const SizedBox.shrink();
    return DecoratedBox(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: c.accent.withValues(alpha: 0.55), width: 1),
        boxShadow: [BoxShadow(color: c.accentGlow, blurRadius: 16)],
      ),
      child: _buildPackCard(c, pack),
    );
  }

  /// Single pack card. Renders differently depending on whether the pack
  /// is currently added:
  ///   • added → whole card tappable (opens the Library reader); trash
  ///     icon on the right.
  ///   • not added → "Add" button on the right; the rest of the card just
  ///     shows pack metadata.
  /// Progress bar swaps in for both states while the add is in flight.
  Widget _buildPackCard(AshColors c, Pack pack) {
    final progress = _progress[pack.id];
    final adding = progress != null;
    final added = pack.installed;
    final canOpen = added && widget.onOpenPack != null;

    final card = Glass(
      radius: 16,
      padding: const EdgeInsets.all(12),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          PackArt(
            size: 48,
            radius: 12,
            fromColor: pack.artFrom,
            toColor: pack.artTo,
            glyph: pack.glyph,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Expanded(
                      child: Text(
                        pack.name,
                        style: TextStyle(
                          color: c.text,
                          fontSize: 15,
                          fontWeight: FontWeight.w600,
                        ),
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                    if (added)
                      Padding(
                        padding: const EdgeInsets.only(left: 6),
                        child: Icon(
                          Icons.check_circle,
                          size: 14,
                          color: c.accent,
                        ),
                      ),
                  ],
                ),
                const SizedBox(height: 4),
                Text(
                  '${pack.size} · v${pack.version} · ${pack.entries} entries',
                  style: TextStyle(color: c.textDim, fontSize: 12),
                ),
                const SizedBox(height: 10),
                Row(
                  children: [
                    if (adding)
                      Expanded(
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(8),
                          child: LinearProgressIndicator(
                            value: progress,
                            backgroundColor: c.surface,
                            valueColor: AlwaysStoppedAnimation(c.accent),
                            minHeight: 6,
                          ),
                        ),
                      )
                    else if (added) ...[
                      Text(
                        'Tap to read',
                        style: TextStyle(color: c.textDim, fontSize: 12),
                      ),
                      const Spacer(),
                      GestureDetector(
                        onTap: () => _removePack(pack),
                        behavior: HitTestBehavior.opaque,
                        child: Padding(
                          padding: const EdgeInsets.all(4),
                          child: Icon(
                            Icons.delete_outline,
                            size: 18,
                            color: c.textMuted,
                          ),
                        ),
                      ),
                    ] else ...[
                      GestureDetector(
                        onTap: () => _addPack(pack),
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 16, vertical: 6),
                          decoration: BoxDecoration(
                            color: c.accentSoft,
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: Text(
                            'Add',
                            style: TextStyle(
                              color: c.accent,
                              fontSize: 13,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                      ),
                      const Spacer(),
                      Text(
                        '${pack.entries} entries',
                        style: TextStyle(color: c.textDim, fontSize: 12),
                      ),
                    ],
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );

    if (!canOpen) return card;
    return GestureDetector(
      onTap: () => widget.onOpenPack!(pack.id),
      behavior: HitTestBehavior.opaque,
      child: card,
    );
  }
}

/// Storage-usage chip in the header.
class _StorageChip extends StatelessWidget {
  const _StorageChip({required this.c, required this.bytes});
  final AshColors c;
  final double bytes;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: c.surfaceElev,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: c.border, width: 0.5),
      ),
      child: Text(
        '${formatBytes(bytes)} used',
        style: TextStyle(
          color: c.textMuted,
          fontSize: 11,
          fontWeight: FontWeight.w600,
        ),
      ),
    );
  }
}

/// Two-state filter chip. Selected variant uses the accent treatment.
class _FilterChip extends StatelessWidget {
  const _FilterChip({
    required this.c,
    required this.label,
    required this.count,
    required this.selected,
    required this.onTap,
  });

  final AshColors c;
  final String label;
  final int count;
  final bool selected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      behavior: HitTestBehavior.opaque,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 150),
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: selected ? c.accentSoft : c.surfaceElev,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(
            color: selected ? c.accent : c.border,
            width: selected ? 1 : 0.5,
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              label,
              style: TextStyle(
                color: selected ? c.accent : c.text,
                fontSize: 13,
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(width: 6),
            Text(
              '$count',
              style: TextStyle(
                color: selected ? c.accent : c.textDim,
                fontSize: 11,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Recommended-row header for the semantic-search overlay.
class _RecommendedHeader extends StatelessWidget {
  const _RecommendedHeader({required this.c});
  final AshColors c;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Icon(Icons.auto_awesome, color: c.accent, size: 14),
            const SizedBox(width: 6),
            Text(
              'RECOMMENDED FOR YOU',
              style: TextStyle(
                color: c.accent,
                fontSize: 11,
                fontWeight: FontWeight.w700,
                letterSpacing: 0.8,
              ),
            ),
          ],
        ),
        const SizedBox(height: 2),
        Text(
          'On-device semantic match for your search.',
          style: TextStyle(color: c.textDim, fontSize: 12),
        ),
      ],
    );
  }
}

/// Tier section header (Essential / Standard / Comprehensive).
class _TierHeader extends StatelessWidget {
  const _TierHeader({
    required this.c,
    required this.tier,
    required this.count,
    required this.showBlurb,
  });

  final AshColors c;
  final PackTier tier;
  final int count;
  final bool showBlurb;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Text(
              tier.label.toUpperCase(),
              style: TextStyle(
                color: c.textDim,
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
        ),
        if (showBlurb) ...[
          const SizedBox(height: 2),
          Text(
            tier.blurb,
            style: TextStyle(color: c.textDim, fontSize: 12),
          ),
        ],
      ],
    );
  }
}
