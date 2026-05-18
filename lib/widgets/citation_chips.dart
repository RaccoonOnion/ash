import 'package:flutter/material.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';

import '../services/chunk_sanitizer.dart';
import '../services/inference_service.dart';
import '../theme/ash_theme.dart';
import 'glass.dart';

/// Horizontal scroll of pill-shaped citation chips, one per retrieval source
/// that grounded the assistant's reply. Tapping a chip opens a bottom sheet
/// with the full chunk text. Hidden when [sources] is null or empty.
class CitationChips extends StatelessWidget {
  const CitationChips({
    required this.sources,
    this.onOpenInLibrary,
    super.key,
  });

  /// The reference chunks. [MessageSource.id] is the `[N]` label used by the
  /// model and on the chip.
  final List<MessageSource> sources;

  /// Optional callback fired from the source-detail sheet's "Open in Library"
  /// button. Receives the packId and the chunkId of the tapped source so the
  /// reader can scroll directly to the cited proposition. `chunkId` may be
  /// empty for sources persisted from older builds — the reader falls back
  /// to opening at the top of the pack in that case. If null, the button is
  /// hidden.
  final void Function(String packId, String chunkId)? onOpenInLibrary;

  @override
  Widget build(BuildContext context) {
    if (sources.isEmpty) return const SizedBox.shrink();
    final c = ash(context);
    return SizedBox(
      height: 26,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        padding: EdgeInsets.zero,
        itemCount: sources.length,
        separatorBuilder: (_, __) => const SizedBox(width: 6),
        itemBuilder: (context, i) {
          final src = sources[i];
          // Surface the chunk's own first heading in the chip — much more
          // informative than "Response › For Adults". Falls back to just
          // the pack name when no heading is extractable.
          final heading = extractChunkHeading(src.text);
          final label = heading?.isNotEmpty == true
              ? '[${src.id}] ${src.packName} · $heading'
              : '[${src.id}] ${src.packName}';
          return _CitationChip(
            color: c,
            label: label,
            onTap: () => _openDetail(context, src),
          );
        },
      ),
    );
  }

  void _openDetail(BuildContext context, MessageSource src) {
    showModalBottomSheet<void>(
      context: context,
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      builder: (sheetContext) => _SourceDetailSheet(
        source: src,
        onOpenInLibrary: onOpenInLibrary == null
            ? null
            : () {
                Navigator.of(sheetContext).pop();
                onOpenInLibrary!(src.packId, src.chunkId);
              },
      ),
    );
  }
}

class _CitationChip extends StatelessWidget {
  const _CitationChip({
    required this.color,
    required this.label,
    required this.onTap,
  });

  final AshColors color;
  final String label;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 4),
        decoration: BoxDecoration(
          color: color.accentSoft,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: color.border, width: 0.5),
        ),
        child: Text(
          label,
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
          style: TextStyle(
            color: color.accent,
            fontSize: 11,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
    );
  }
}

class _SourceDetailSheet extends StatelessWidget {
  const _SourceDetailSheet({
    required this.source,
    required this.onOpenInLibrary,
  });

  final MessageSource source;
  final VoidCallback? onOpenInLibrary;

  @override
  Widget build(BuildContext context) {
    final c = ash(context);
    final mq = MediaQuery.of(context);
    final parts = parseSectionPath(source.sectionPath);
    final heading = extractChunkHeading(source.text);
    // Strip the chunk's first heading from the body so the title above
    // doesn't render again inline.
    final body = heading != null ? stripFirstHeading(source.text) : source.text;
    return SafeArea(
      child: Padding(
        padding: EdgeInsets.fromLTRB(
          16,
          8,
          16,
          16 + mq.viewInsets.bottom,
        ),
        child: Glass(
          radius: 24,
          strong: true,
          padding: const EdgeInsets.fromLTRB(18, 14, 18, 14),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Drag handle
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
              // [N] PackName · ExtractedHeading
              Row(
                children: [
                  Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                    decoration: BoxDecoration(
                      color: c.accent,
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(
                      '[${source.id}]',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 11,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text(
                          heading?.isNotEmpty == true
                              ? heading!
                              : source.packName,
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                          style: TextStyle(
                            color: c.text,
                            fontSize: 15,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                        if (heading?.isNotEmpty == true)
                          Text(
                            source.packName,
                            style:
                                TextStyle(color: c.textMuted, fontSize: 12),
                          ),
                      ],
                    ),
                  ),
                ],
              ),
              if (parts.phase != null || parts.tail.isNotEmpty) ...[
                const SizedBox(height: 8),
                Padding(
                  padding: const EdgeInsets.only(left: 36),
                  child: Wrap(
                    spacing: 6,
                    runSpacing: 6,
                    children: [
                      if (parts.phase != null)
                        _SheetBadge(
                          c: c,
                          label: parts.phase!,
                          accent: true,
                        ),
                      if (parts.tail.isNotEmpty)
                        _SheetBadge(c: c, label: parts.tail),
                    ],
                  ),
                ),
              ],
              const SizedBox(height: 14),
              // Chunk text. Cap height so the sheet stays scrollable for
              // long sections instead of taking over the screen.
              ConstrainedBox(
                constraints: BoxConstraints(maxHeight: mq.size.height * 0.45),
                child: Scrollbar(
                  child: SingleChildScrollView(
                    child: MarkdownBody(
                      data: body,
                      selectable: true,
                      softLineBreak: true,
                      styleSheet: MarkdownStyleSheet(
                        p: TextStyle(
                          color: c.text,
                          fontSize: 13.5,
                          height: 1.45,
                        ),
                        strong: TextStyle(
                          color: c.text,
                          fontWeight: FontWeight.w700,
                        ),
                        listBullet: TextStyle(color: c.accent),
                      ),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 14),
              if (onOpenInLibrary != null)
                Row(
                  children: [
                    Expanded(
                      child: GestureDetector(
                        onTap: onOpenInLibrary,
                        child: Container(
                          height: 40,
                          alignment: Alignment.center,
                          decoration: BoxDecoration(
                            color: c.accentSoft,
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: Text(
                            'Open in Library',
                            style: TextStyle(
                              color: c.accent,
                              fontSize: 13,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
            ],
          ),
        ),
      ),
    );
  }
}

/// Tiny chip used inside the source-detail sheet for the phase / audience
/// labels (mirrors the Library reader's badges but rendered here as well so
/// the citation view stands on its own).
class _SheetBadge extends StatelessWidget {
  const _SheetBadge({
    required this.c,
    required this.label,
    this.accent = false,
  });

  final AshColors c;
  final String label;
  final bool accent;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: accent ? c.accent : c.surfaceElev,
        borderRadius: BorderRadius.circular(8),
        border: accent ? null : Border.all(color: c.border, width: 0.5),
      ),
      child: Text(
        label,
        style: TextStyle(
          color: accent ? Colors.white : c.textMuted,
          fontSize: 10,
          fontWeight: FontWeight.w600,
          letterSpacing: 0.3,
        ),
      ),
    );
  }
}
