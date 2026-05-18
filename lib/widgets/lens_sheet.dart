import 'package:flutter/material.dart' hide Chip;
import '../models/pack.dart';
import '../theme/ash_theme.dart';
import '../utils/fuzzy.dart';
import 'glass.dart';
import 'text_field.dart';

/// Bottom sheet that lets the user scope a chat's retrieval lens to a subset
/// of installed packs (or back to "all installed"). Returned via
/// [Navigator.pop] as a [LensResult] when the user taps Done; null otherwise.
///
/// The selection rules:
///   • Empty selection ⇒ returned as `null` (= retrieve from all installed)
///   • Tapping "Use all" clears the selection
///   • Tapping a row toggles that pack in/out of the selection
class LensSheet extends StatefulWidget {
  const LensSheet({
    required this.installedPacks,
    required this.initialLens,
    super.key,
  });

  final List<Pack> installedPacks;
  final Set<String>? initialLens;

  @override
  State<LensSheet> createState() => _LensSheetState();
}

class _LensSheetState extends State<LensSheet> {
  late Set<String> _selected;
  final _queryController = TextEditingController();
  String _query = '';

  @override
  void initState() {
    super.initState();
    _selected = widget.initialLens?.toSet() ?? <String>{};
  }

  @override
  void dispose() {
    _queryController.dispose();
    super.dispose();
  }

  bool get _isAll => _selected.isEmpty;

  List<Pack> get _filtered {
    final q = _query.trim();
    if (q.isEmpty) return widget.installedPacks;
    final scored = <({Pack pack, int score})>[];
    for (final p in widget.installedPacks) {
      final s = fuzzyScoreFields(q, [
        (text: p.name, weight: 1.0),
        (text: p.summary, weight: 0.5),
      ]);
      if (s == null) continue;
      scored.add((pack: p, score: s));
    }
    scored.sort((a, b) => b.score.compareTo(a.score));
    return scored.map((e) => e.pack).toList();
  }

  void _commit() {
    Navigator.of(context).pop(
      LensResult(lensPackIds: _isAll ? null : _selected.toSet()),
    );
  }

  @override
  Widget build(BuildContext context) {
    final c = ash(context);
    final mq = MediaQuery.of(context);
    final filtered = _filtered;
    return SafeArea(
      child: Padding(
        padding: EdgeInsets.fromLTRB(16, 8, 16, 16 + mq.viewInsets.bottom),
        child: Glass(
          radius: 24,
          strong: true,
          padding: const EdgeInsets.fromLTRB(18, 14, 18, 14),
          child: SizedBox(
            // Roughly half-screen; the inner list scrolls.
            height: mq.size.height * 0.65,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Center(
                  child: Container(
                    width: 36,
                    height: 4,
                    decoration: BoxDecoration(
                      color: c.textDim,
                      borderRadius: BorderRadius.circular(2),
                    ),
                  ),
                ),
                const SizedBox(height: 14),
                Row(
                  children: [
                    Icon(Icons.menu_book_outlined,
                        color: c.accent, size: 18),
                    const SizedBox(width: 8),
                    Text(
                      'Knowledge lens',
                      style: TextStyle(
                        color: c.text,
                        fontSize: 16,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    const Spacer(),
                    GestureDetector(
                      onTap: () => setState(_selected.clear),
                      child: Text(
                        'Use all',
                        style: TextStyle(
                          color: _isAll ? c.textDim : c.accent,
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 6),
                Text(
                  _isAll
                      ? 'Retrieving from all ${widget.installedPacks.length} '
                          'installed packs. Pick specific packs to scope down.'
                      : '${_selected.length} of ${widget.installedPacks.length} '
                          'pack${_selected.length == 1 ? '' : 's'} selected.',
                  style: TextStyle(color: c.textMuted, fontSize: 12),
                ),
                const SizedBox(height: 12),
                AshTextField(
                  hint: 'Filter packs',
                  controller: _queryController,
                  leadingIcon: Icons.search,
                  onChanged: (v) => setState(() => _query = v),
                ),
                const SizedBox(height: 10),
                Expanded(
                  child: filtered.isEmpty
                      ? Center(
                          child: Text(
                            'No packs match "$_query".',
                            style: TextStyle(color: c.textDim, fontSize: 13),
                          ),
                        )
                      : ListView.separated(
                          itemCount: filtered.length,
                          separatorBuilder: (_, __) => const SizedBox(height: 6),
                          itemBuilder: (context, i) {
                            final p = filtered[i];
                            final picked = _selected.contains(p.id);
                            return _LensRow(
                              c: c,
                              pack: p,
                              picked: picked,
                              onTap: () => setState(() {
                                if (picked) {
                                  _selected.remove(p.id);
                                } else {
                                  _selected.add(p.id);
                                }
                              }),
                            );
                          },
                        ),
                ),
                const SizedBox(height: 10),
                GestureDetector(
                  onTap: _commit,
                  child: Container(
                    height: 44,
                    alignment: Alignment.center,
                    decoration: BoxDecoration(
                      color: c.accent,
                      borderRadius: BorderRadius.circular(14),
                    ),
                    child: Text(
                      _isAll
                          ? 'Use all packs'
                          : 'Use ${_selected.length} pack'
                              '${_selected.length == 1 ? '' : 's'}',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _LensRow extends StatelessWidget {
  const _LensRow({
    required this.c,
    required this.pack,
    required this.picked,
    required this.onTap,
  });

  final AshColors c;
  final Pack pack;
  final bool picked;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      behavior: HitTestBehavior.opaque,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        decoration: BoxDecoration(
          color: picked ? c.accentSoft : c.surfaceElev,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: picked ? c.accent : c.border,
            width: picked ? 1 : 0.5,
          ),
        ),
        child: Row(
          children: [
            Icon(
              picked ? Icons.check_circle : Icons.circle_outlined,
              size: 18,
              color: picked ? c.accent : c.textDim,
            ),
            const SizedBox(width: 10),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    pack.name,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(
                      color: c.text,
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  Text(
                    '${pack.entries} entries · ${pack.size}',
                    style: TextStyle(color: c.textDim, fontSize: 11),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Wrapped return value so we can distinguish "no selection (null)" from
/// "sheet dismissed without committing (Navigator pop with no value)".
class LensResult {
  const LensResult({required this.lensPackIds});
  final Set<String>? lensPackIds;
}
