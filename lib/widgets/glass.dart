import 'dart:ui';
import 'package:flutter/material.dart';
import '../theme/ash_theme.dart';

/// A glass-morphism container that applies backdrop-blur + tinted background.
///
/// Mirrors the HTML prototype's `<Glass>` component.
class Glass extends StatelessWidget {
  const Glass({
    super.key,
    required this.child,
    this.radius = 22,
    this.padding = EdgeInsets.zero,
    this.strong = false,
    this.blur = 22,
    this.color,
    this.borderColor,
    this.shadow,
  });

  final Widget child;

  /// Border radius of the container. Default 22.
  final double radius;

  /// Inner padding around [child].
  final EdgeInsetsGeometry padding;

  /// When true, uses `glassTintStrong` instead of `glassTint`.
  final bool strong;

  /// Sigma value for the [BackdropFilter]. Default 22.
  final double blur;

  /// Override the background tint colour.
  final Color? color;

  /// Override the border colour.
  final Color? borderColor;

  /// Override the box shadow.
  final List<BoxShadow>? shadow;

  @override
  Widget build(BuildContext context) {
    final c = ash(context);

    final bgColor = color ??
        (strong ? c.glassTintStrong : c.glassTint);
    final bdColor = borderColor ?? c.border;

    return ClipRRect(
      borderRadius: BorderRadius.circular(radius),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: blur, sigmaY: blur),
        child: Container(
          padding: padding,
          decoration: BoxDecoration(
            color: bgColor,
            borderRadius: BorderRadius.circular(radius),
            border: Border.all(color: bdColor, width: 0.5),
            boxShadow: shadow ??
                [
                  BoxShadow(
                    color: Colors.black.withValues(alpha: 0.25),
                    blurRadius: 16,
                    offset: const Offset(0, 4),
                  ),
                ],
          ),
          child: child,
        ),
      ),
    );
  }
}
