import 'dart:math' as math;
import 'package:flutter/material.dart';
import '../theme/ash_theme.dart';

/// Gradient square artwork for knowledge packs with an optional topo-line
/// overlay and glyph text.
class PackArt extends StatelessWidget {
  const PackArt({
    super.key,
    this.size = 56,
    this.radius = 12,
    this.fromColor,
    this.toColor,
    this.glyph,
  });

  final double size;
  final double radius;
  final Color? fromColor;
  final Color? toColor;
  final String? glyph;

  @override
  Widget build(BuildContext context) {
    final c = ash(context);

    final gradFrom = fromColor ?? c.accent.withValues(alpha: 0.7);
    final gradTo = toColor ?? c.surfaceElev;

    return SizedBox(
      width: size,
      height: size,
      child: CustomPaint(
        painter: _PackArtPainter(
          fromColor: gradFrom,
          toColor: gradTo,
          radius: radius,
          lineColor: Colors.white.withValues(alpha: 0.07),
        ),
        child: Stack(
          children: [
            if (glyph != null)
              Positioned(
                top: 4,
                left: 5,
                child: Text(
                  glyph!,
                  style: TextStyle(
                    color: Colors.white.withValues(alpha: 0.55),
                    fontSize: 9,
                    fontFamily: 'monospace',
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _PackArtPainter extends CustomPainter {
  _PackArtPainter({
    required this.fromColor,
    required this.toColor,
    required this.radius,
    required this.lineColor,
  });

  final Color fromColor;
  final Color toColor;
  final double radius;
  final Color lineColor;

  @override
  void paint(Canvas canvas, Size size) {
    final rect = Offset.zero & size;
    final rrect = RRect.fromRectAndRadius(rect, Radius.circular(radius));

    // Gradient background (135 degrees)
    final gradient = LinearGradient(
      begin: Alignment.topLeft,
      end: Alignment.bottomRight,
      colors: [fromColor, toColor],
    );
    final paint = Paint()..shader = gradient.createShader(rect);
    canvas.drawRRect(rrect, paint);

    // Subtle topo-line overlay (3 wavy horizontal lines)
    final linePaint = Paint()
      ..color = lineColor
      ..style = PaintingStyle.stroke
      ..strokeWidth = 0.8;

    for (var i = 1; i <= 3; i++) {
      final y = size.height * i / 4;
      final path = Path();
      path.moveTo(0, y);
      for (double x = 0; x <= size.width; x += 4) {
        path.lineTo(
          x,
          y + 2.0 * math.sin(x * 0.15 + i),
        );
      }
      canvas.drawPath(path, linePaint);
    }
  }

  @override
  bool shouldRepaint(covariant _PackArtPainter old) =>
      fromColor != old.fromColor ||
      toColor != old.toColor ||
      radius != old.radius;
}
