import 'package:flutter/material.dart';
import '../theme/ash_theme.dart';

// ─── Chip ────────────────────────────────────────────────────────────────────

/// Small inline badge with accent or muted styling.
enum ChipVariant { accent, muted }

class Chip extends StatelessWidget {
  const Chip({
    super.key,
    required this.label,
    this.variant = ChipVariant.accent,
  });

  final String label;
  final ChipVariant variant;

  @override
  Widget build(BuildContext context) {
    final c = ash(context);

    final isAccent = variant == ChipVariant.accent;
    final bgColor = isAccent ? c.accentSoft : c.surface;
    final textColor = isAccent ? c.accent : c.textMuted;
    final bdColor = isAccent ? Colors.transparent : c.border;

    return Container(
      height: 24,
      padding: const EdgeInsets.symmetric(horizontal: 8),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: bdColor, width: 0.5),
      ),
      child: Center(
        child: Text(
          label.toUpperCase(),
          style: TextStyle(
            color: textColor,
            fontSize: 11.5,
            fontWeight: FontWeight.w600,
            letterSpacing: 0.5,
          ),
        ),
      ),
    );
  }
}

// ─── SectionLabel ────────────────────────────────────────────────────────────

/// 11 px uppercase section heading with textDim colour.
class SectionLabel extends StatelessWidget {
  const SectionLabel({
    super.key,
    required this.label,
  });

  final String label;

  @override
  Widget build(BuildContext context) {
    final c = ash(context);

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Text(
        label.toUpperCase(),
        style: TextStyle(
          color: c.textDim,
          fontSize: 11,
          fontWeight: FontWeight.w600,
          letterSpacing: 0.8,
        ),
      ),
    );
  }
}
