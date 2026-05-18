import 'package:flutter/material.dart';

/// Ash color tokens – dark theme only.
///
/// Every token matches the HTML prototype's theme.jsx exactly.
@immutable
class AshColors extends ThemeExtension<AshColors> {
  const AshColors({
    required this.bg,
    required this.surface,
    required this.surfaceElev,
    required this.surfaceStrong,
    required this.glassTint,
    required this.glassTintStrong,
    required this.border,
    required this.borderStrong,
    required this.text,
    required this.textMuted,
    required this.textDim,
    required this.accent,
    required this.accentSoft,
    required this.accentGlow,
    required this.success,
  });

  // ── Background ──────────────────────────────────────────────
  final Color bg;

  // ── Surfaces ────────────────────────────────────────────────
  final Color surface; // rgba(26,26,26,0.85)
  final Color surfaceElev; // rgba(38,38,38,0.92)
  final Color surfaceStrong; // rgba(26,26,26,0.92)

  // ── Glass tints ─────────────────────────────────────────────
  final Color glassTint; // rgba(38,38,38,0.65)
  final Color glassTintStrong; // rgba(51,51,51,0.75)

  // ── Borders ─────────────────────────────────────────────────
  final Color border; // rgba(255,255,255,0.06)
  final Color borderStrong; // rgba(255,255,255,0.12)

  // ── Text ────────────────────────────────────────────────────
  final Color text; // #F0EDE8
  final Color textMuted; // #B8B4AE
  final Color textDim; // #6B6865

  // ── Accent ──────────────────────────────────────────────────
  final Color accent; // #CC3F1E
  final Color accentSoft; // rgba(204,63,30,0.14)
  final Color accentGlow; // rgba(204,63,30,0.40)

  // ── Semantic ────────────────────────────────────────────────
  final Color success; // #4CAF50

  // ── Dark instance (the only theme) ──────────────────────────
  static const AshColors dark = AshColors(
    bg: Color(0xFF0F0F0F),
    surface: Color.fromARGB(217, 26, 26, 26),
    surfaceElev: Color.fromARGB(235, 38, 38, 38),
    surfaceStrong: Color.fromARGB(235, 26, 26, 26),
    glassTint: Color.fromARGB(166, 38, 38, 38),
    glassTintStrong: Color.fromARGB(191, 51, 51, 51),
    border: Color.fromARGB(15, 255, 255, 255),
    borderStrong: Color.fromARGB(31, 255, 255, 255),
    text: Color(0xFFF0EDE8),
    textMuted: Color(0xFFB8B4AE),
    textDim: Color(0xFF6B6865),
    accent: Color(0xFFCC3F1E),
    accentSoft: Color.fromARGB(36, 204, 63, 30),
    accentGlow: Color.fromARGB(102, 204, 63, 30),
    success: Color(0xFF4CAF50),
  );

  @override
  AshColors copyWith({
    Color? bg,
    Color? surface,
    Color? surfaceElev,
    Color? surfaceStrong,
    Color? glassTint,
    Color? glassTintStrong,
    Color? border,
    Color? borderStrong,
    Color? text,
    Color? textMuted,
    Color? textDim,
    Color? accent,
    Color? accentSoft,
    Color? accentGlow,
    Color? success,
  }) {
    return AshColors(
      bg: bg ?? this.bg,
      surface: surface ?? this.surface,
      surfaceElev: surfaceElev ?? this.surfaceElev,
      surfaceStrong: surfaceStrong ?? this.surfaceStrong,
      glassTint: glassTint ?? this.glassTint,
      glassTintStrong: glassTintStrong ?? this.glassTintStrong,
      border: border ?? this.border,
      borderStrong: borderStrong ?? this.borderStrong,
      text: text ?? this.text,
      textMuted: textMuted ?? this.textMuted,
      textDim: textDim ?? this.textDim,
      accent: accent ?? this.accent,
      accentSoft: accentSoft ?? this.accentSoft,
      accentGlow: accentGlow ?? this.accentGlow,
      success: success ?? this.success,
    );
  }

  @override
  AshColors lerp(AshColors? other, double t) {
    if (other is! AshColors) return this;
    return AshColors(
      bg: Color.lerp(bg, other.bg, t)!,
      surface: Color.lerp(surface, other.surface, t)!,
      surfaceElev: Color.lerp(surfaceElev, other.surfaceElev, t)!,
      surfaceStrong: Color.lerp(surfaceStrong, other.surfaceStrong, t)!,
      glassTint: Color.lerp(glassTint, other.glassTint, t)!,
      glassTintStrong: Color.lerp(glassTintStrong, other.glassTintStrong, t)!,
      border: Color.lerp(border, other.border, t)!,
      borderStrong: Color.lerp(borderStrong, other.borderStrong, t)!,
      text: Color.lerp(text, other.text, t)!,
      textMuted: Color.lerp(textMuted, other.textMuted, t)!,
      textDim: Color.lerp(textDim, other.textDim, t)!,
      accent: Color.lerp(accent, other.accent, t)!,
      accentSoft: Color.lerp(accentSoft, other.accentSoft, t)!,
      accentGlow: Color.lerp(accentGlow, other.accentGlow, t)!,
      success: Color.lerp(success, other.success, t)!,
    );
  }
}

/// Shortcut to read [AshColors] from the current [BuildContext].
AshColors ash(BuildContext context) {
  return Theme.of(context).extension<AshColors>() ?? AshColors.dark;
}
