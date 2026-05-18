import 'package:flutter/material.dart';
import '../theme/ash_theme.dart';

// ─── PrimaryBtn ──────────────────────────────────────────────────────────────

/// Full-width ember-accent button with glow shadow.
class PrimaryBtn extends StatelessWidget {
  const PrimaryBtn({
    super.key,
    required this.label,
    this.onTap,
    this.enabled = true,
    this.height = 52,
    this.borderRadius = 26,
  });

  final String label;
  final VoidCallback? onTap;
  final bool enabled;
  final double height;
  final double borderRadius;

  @override
  Widget build(BuildContext context) {
    final c = ash(context);

    final bgColor = enabled ? c.accent : c.surfaceStrong;
    final textColor = enabled ? Colors.white : c.textDim;
    final shadowColor = enabled ? c.accentGlow : Colors.transparent;

    return SizedBox(
      width: double.infinity,
      height: height,
      child: DecoratedBox(
        decoration: BoxDecoration(
          gradient: enabled
              ? LinearGradient(
                  colors: [c.accent, c.accent],
                )
              : null,
          color: enabled ? null : bgColor,
          borderRadius: BorderRadius.circular(borderRadius),
          boxShadow: [
            BoxShadow(
              color: shadowColor,
              blurRadius: 18,
              offset: const Offset(0, 6),
            ),
          ],
        ),
        child: Material(
          color: Colors.transparent,
          child: InkWell(
            onTap: enabled ? onTap : null,
            borderRadius: BorderRadius.circular(borderRadius),
            child: Center(
              child: Text(
                label,
                style: TextStyle(
                  color: textColor,
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

// ─── GhostBtn ────────────────────────────────────────────────────────────────

/// Glass secondary button with surface background and subtle border.
class GhostBtn extends StatelessWidget {
  const GhostBtn({
    super.key,
    required this.label,
    this.onTap,
    this.height = 50,
    this.borderRadius = 25,
  });

  final String label;
  final VoidCallback? onTap;
  final double height;
  final double borderRadius;

  @override
  Widget build(BuildContext context) {
    final c = ash(context);

    return SizedBox(
      height: height,
      child: DecoratedBox(
        decoration: BoxDecoration(
          color: c.surface,
          borderRadius: BorderRadius.circular(borderRadius),
          border: Border.all(color: c.border, width: 0.5),
        ),
        child: Material(
          color: Colors.transparent,
          child: InkWell(
            onTap: onTap,
            borderRadius: BorderRadius.circular(borderRadius),
            child: Center(
              child: Text(
                label,
                style: TextStyle(
                  color: c.text,
                  fontSize: 15,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

// ─── IconBtn ─────────────────────────────────────────────────────────────────

/// Round glass icon button with surface or accent background.
class IconBtn extends StatelessWidget {
  const IconBtn({
    super.key,
    required this.icon,
    this.onTap,
    this.size = 36,
    this.accentBg = false,
  });

  final IconData icon;
  final VoidCallback? onTap;
  final double size;
  final bool accentBg;

  @override
  Widget build(BuildContext context) {
    final c = ash(context);

    final bgColor = accentBg ? c.accent : c.surface;
    final iconColor = accentBg ? Colors.white : c.text;

    return SizedBox(
      width: size,
      height: size,
      child: DecoratedBox(
        decoration: BoxDecoration(
          color: bgColor,
          shape: BoxShape.circle,
          border: accentBg
              ? null
              : Border.all(color: c.border, width: 0.5),
        ),
        child: Material(
          color: Colors.transparent,
          child: InkWell(
            onTap: onTap,
            customBorder: const CircleBorder(),
            child: Icon(
              icon,
              size: size * 0.44,
              color: iconColor,
            ),
          ),
        ),
      ),
    );
  }
}
