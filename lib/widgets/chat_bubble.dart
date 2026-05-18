import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter_markdown_plus/flutter_markdown_plus.dart';
import '../services/inference_service.dart';
import '../theme/ash_theme.dart';
import 'ash_mark.dart';
import 'citation_chips.dart';

// ─── UserBubble ─────────────────────────────────────────────────────────────

/// Right-aligned accent gradient bubble for user messages.
class UserBubble extends StatelessWidget {
  const UserBubble({required this.text, this.imageBytes, super.key});

  final String text;
  final Uint8List? imageBytes;

  @override
  Widget build(BuildContext context) {
    final c = ash(context);
    final hasImage = imageBytes != null;

    return Align(
      alignment: Alignment.centerRight,
      child: ConstrainedBox(
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.82,
        ),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topCenter,
              end: Alignment.bottomCenter,
              colors: [c.accent, c.accent],
            ),
            borderRadius: const BorderRadius.only(
              topLeft: Radius.circular(22),
              topRight: Radius.circular(22),
              bottomLeft: Radius.circular(22),
              bottomRight: Radius.circular(8),
            ),
            boxShadow: [
              BoxShadow(
                color: c.accentGlow,
                blurRadius: 12,
                offset: const Offset(0, 4),
              ),
            ],
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.end,
            mainAxisSize: MainAxisSize.min,
            children: [
              if (hasImage)
                ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: Image.memory(
                    imageBytes!,
                    width: double.infinity,
                    fit: BoxFit.cover,
                    errorBuilder: (_, __, ___) => Container(
                      height: 120,
                      color: Colors.white24,
                      child: const Center(
                        child: Icon(Icons.broken_image, color: Colors.white54),
                      ),
                    ),
                  ),
                ),
              if (hasImage) const SizedBox(height: 8),
              Text(
                text,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 15,
                  height: 1.4,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ─── AssistantBubble ────────────────────────────────────────────────────────

/// Parsed structure of an assistant message that may contain a thinking
/// trace. When the model runs with `isThinking: true`, it emits its
/// reasoning inside `<think>...</think>` before the final answer.
class _ThoughtSplit {
  const _ThoughtSplit({
    required this.thinking,
    required this.answer,
    required this.thinkingOpen,
  });
  final String thinking;
  final String answer;
  /// True when we've seen `<think>` but not yet the closing tag — i.e.
  /// the model is still mid-thought. Lets the UI show "Thinking…".
  final bool thinkingOpen;
}

_ThoughtSplit _splitThought(String raw) {
  final openIdx = raw.indexOf('<think>');
  if (openIdx < 0) {
    return _ThoughtSplit(
      thinking: '',
      answer: raw,
      thinkingOpen: false,
    );
  }
  final closeIdx = raw.indexOf('</think>', openIdx);
  final before = raw.substring(0, openIdx);
  if (closeIdx < 0) {
    // Mid-stream: opened but not yet closed.
    final thinking = raw.substring(openIdx + 7).trim();
    return _ThoughtSplit(
      thinking: thinking,
      answer: before.trim(),
      thinkingOpen: true,
    );
  }
  final thinking = raw.substring(openIdx + 7, closeIdx).trim();
  final after = raw.substring(closeIdx + 8);
  return _ThoughtSplit(
    thinking: thinking,
    answer: (before + after).trim(),
    thinkingOpen: false,
  );
}

/// Left-aligned glass bubble for assistant messages. TTS playback is no
/// longer per-message — use Live voice mode (mic button) to hear replies
/// read aloud as they stream.
///
/// When the service grounded the reply on retrieved chunks, [sources] is
/// populated and a row of [CitationChips] renders below the bubble. The
/// chips link back into the Library reader via [onOpenInLibrary] when
/// provided.
///
/// When the chat has thinking mode enabled, the message text may include
/// a `<think>...</think>` block. We surface it as a small collapsible
/// pill above the main answer rather than rendering it inline.
class AssistantBubble extends StatefulWidget {
  const AssistantBubble({
    required this.text,
    this.streaming = false,
    this.sources,
    this.onOpenInLibrary,
    super.key,
  });

  final String text;
  final bool streaming;
  final List<MessageSource>? sources;
  final void Function(String packId, String chunkId)? onOpenInLibrary;

  @override
  State<AssistantBubble> createState() => _AssistantBubbleState();
}

class _AssistantBubbleState extends State<AssistantBubble> {
  bool _thoughtExpanded = false;

  @override
  Widget build(BuildContext context) {
    final c = ash(context);
    final split = _splitThought(widget.text);
    // When the model is still mid-thought (open `<think>` but no close
    // tag), the answer body will be empty — show only the "Thinking…"
    // pill until the reply lands.
    final hasThought = split.thinking.isNotEmpty || split.thinkingOpen;

    return Align(
      alignment: Alignment.centerLeft,
      child: ConstrainedBox(
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.88,
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (hasThought) ...[
              Padding(
                padding: const EdgeInsets.only(left: 36, bottom: 6),
                child: _ThoughtPill(
                  c: c,
                  thinking: split.thinking,
                  isOpen: split.thinkingOpen,
                  expanded: _thoughtExpanded,
                  onToggle: () => setState(
                      () => _thoughtExpanded = !_thoughtExpanded),
                ),
              ),
            ],
            Row(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                // Avatar circle
                Container(
                  width: 28,
                  height: 28,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: c.surfaceElev,
                    border: Border.all(color: c.border, width: 0.5),
                  ),
                  child: const Center(
                    child: AshMark(size: 16),
                  ),
                ),
                const SizedBox(width: 8),
                // Glass bubble — only renders when there's answer body or
                // the message is non-thinking. While the model is mid-
                // thought, the bubble would be empty and visually noisy,
                // so we hide it until the answer begins.
                if (split.answer.isNotEmpty || !split.thinkingOpen)
                  Flexible(
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 14,
                        vertical: 12,
                      ),
                      decoration: BoxDecoration(
                        color: c.glassTint,
                        borderRadius: const BorderRadius.only(
                          topLeft: Radius.circular(20),
                          topRight: Radius.circular(20),
                          bottomLeft: Radius.circular(8),
                          bottomRight: Radius.circular(20),
                        ),
                        border: Border.all(color: c.border, width: 0.5),
                      ),
                      child: _buildFormattedText(context, split.answer),
                    ),
                  ),
              ],
            ),
            const SizedBox(height: 6),
            Padding(
              padding: const EdgeInsets.only(left: 36),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  if (widget.sources != null && widget.sources!.isNotEmpty) ...[
                    CitationChips(
                      sources: widget.sources!,
                      onOpenInLibrary: widget.onOpenInLibrary,
                    ),
                    const SizedBox(height: 6),
                  ],
                  Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.signal_cellular_alt,
                          size: 11, color: c.textDim),
                      const SizedBox(width: 4),
                      Text(
                        widget.sources != null && widget.sources!.isNotEmpty
                            ? 'Grounded in ${widget.sources!.length} source'
                                '${widget.sources!.length == 1 ? '' : 's'} · '
                                'On-device'
                            : 'On-device',
                        style: TextStyle(color: c.textDim, fontSize: 11),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildFormattedText(BuildContext context, String body) {
    final c = ash(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        if (body.isNotEmpty)
          // RepaintBoundary isolates this subtree from sibling paint passes
          // so the per-chunk rebuilds don't force the whole ListView to
          // repaint — that was the dominant source of streaming ghosts.
          RepaintBoundary(
            child: MarkdownBody(
              data: body,
              // Selectable is expensive (a TextSelectionGestureDetector +
              // an extra paint layer for selection rects) — turn it off
              // mid-stream and only re-enable once the message is settled.
              selectable: !widget.streaming,
              // Treat `\n` as line break (Gemma 4 emits prose without two
              // trailing spaces / blank lines between steps — without this
              // numbered lists collapse onto a single line).
              softLineBreak: true,
              styleSheet: _markdownStyle(c),
            ),
          ),
        if (widget.streaming) _BlinkingCursor(),
      ],
    );
  }

  MarkdownStyleSheet _markdownStyle(AshColors c) {
    final base = TextStyle(color: c.text, fontSize: 14, height: 1.5);
    return MarkdownStyleSheet(
      p: base,
      h1: base.copyWith(fontSize: 20, fontWeight: FontWeight.w700, height: 1.3),
      h2: base.copyWith(fontSize: 18, fontWeight: FontWeight.w700, height: 1.3),
      h3: base.copyWith(fontSize: 16, fontWeight: FontWeight.w700, height: 1.3),
      h4: base.copyWith(fontSize: 14, fontWeight: FontWeight.w700),
      h5: base.copyWith(fontSize: 13, fontWeight: FontWeight.w700),
      h6: base.copyWith(fontSize: 12, fontWeight: FontWeight.w700),
      strong: base.copyWith(fontWeight: FontWeight.w700),
      em: base.copyWith(fontStyle: FontStyle.italic),
      listBullet: base.copyWith(color: c.accent),
      // Inline `code`
      code: TextStyle(
        color: c.accent,
        fontSize: 13,
        fontFamily: 'Menlo',
        backgroundColor: c.surfaceElev.withValues(alpha: 0.6),
      ),
      // ``` fenced code block ```
      codeblockPadding: const EdgeInsets.all(10),
      codeblockDecoration: BoxDecoration(
        color: c.surfaceElev,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: c.border, width: 0.5),
      ),
      blockquoteDecoration: BoxDecoration(
        color: c.surfaceElev.withValues(alpha: 0.4),
        border: Border(left: BorderSide(color: c.accent, width: 3)),
      ),
      blockquotePadding: const EdgeInsets.fromLTRB(10, 6, 8, 6),
      a: base.copyWith(
        color: c.accent,
        decoration: TextDecoration.underline,
      ),
      tableCellsPadding: const EdgeInsets.symmetric(
        horizontal: 8,
        vertical: 4,
      ),
      tableBorder: TableBorder.all(color: c.border, width: 0.5),
      tableHead: base.copyWith(fontWeight: FontWeight.w700),
      horizontalRuleDecoration: BoxDecoration(
        border: Border(top: BorderSide(color: c.border, width: 0.5)),
      ),
    );
  }
}

// ─── Thought pill ───────────────────────────────────────────────────────────

/// Collapsed-by-default summary of the model's `<think>...</think>` block.
/// Tap to expand and read the reasoning trace. Streaming-while-open shows
/// "Thinking…" instead of a static label so the user knows the model is
/// reasoning rather than stalled.
class _ThoughtPill extends StatelessWidget {
  const _ThoughtPill({
    required this.c,
    required this.thinking,
    required this.isOpen,
    required this.expanded,
    required this.onToggle,
  });

  final AshColors c;
  final String thinking;
  final bool isOpen;
  final bool expanded;
  final VoidCallback onToggle;

  @override
  Widget build(BuildContext context) {
    final label = isOpen
        ? 'Thinking…'
        : (expanded ? 'Hide reasoning' : 'Show reasoning');
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        GestureDetector(
          onTap: isOpen ? null : onToggle,
          behavior: HitTestBehavior.opaque,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
            decoration: BoxDecoration(
              color: c.surfaceElev,
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: c.border, width: 0.5),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(
                  isOpen
                      ? Icons.psychology_outlined
                      : (expanded
                          ? Icons.expand_less
                          : Icons.expand_more),
                  size: 14,
                  color: c.textMuted,
                ),
                const SizedBox(width: 6),
                Text(
                  label,
                  style: TextStyle(
                    color: c.textMuted,
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ),
          ),
        ),
        if (expanded && thinking.isNotEmpty) ...[
          const SizedBox(height: 6),
          // Reasoning trace presented as a quote-like aside: a 2px
          // accent left bar (this is the model's "inner voice") and a
          // monospace font (structured logic, not prose). Italic and
          // muted color keep it visually subordinate so the assistant's
          // actual answer remains the primary read.
          Container(
            padding: const EdgeInsets.fromLTRB(12, 8, 10, 8),
            decoration: BoxDecoration(
              color: c.surfaceElev.withValues(alpha: 0.5),
              borderRadius: BorderRadius.circular(10),
              border: Border(
                left: BorderSide(
                  color: c.accent.withValues(alpha: 0.5),
                  width: 2,
                ),
                top: BorderSide(color: c.border, width: 0.5),
                right: BorderSide(color: c.border, width: 0.5),
                bottom: BorderSide(color: c.border, width: 0.5),
              ),
            ),
            child: Text(
              thinking,
              style: TextStyle(
                color: c.textMuted,
                fontSize: 11.5,
                height: 1.45,
                fontStyle: FontStyle.italic,
                fontFamily: 'Menlo',
              ),
            ),
          ),
        ],
      ],
    );
  }
}

// ─── Blinking cursor ────────────────────────────────────────────────────────

class _BlinkingCursor extends StatefulWidget {
  @override
  State<_BlinkingCursor> createState() => _BlinkingCursorState();
}

class _BlinkingCursorState extends State<_BlinkingCursor>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 600),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final c = ash(context);
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, _) {
        return Opacity(
          opacity: _controller.value,
          child: Text(
            '|',
            style: TextStyle(
              color: c.accent,
              fontSize: 14,
              fontWeight: FontWeight.w700,
            ),
          ),
        );
      },
    );
  }
}
