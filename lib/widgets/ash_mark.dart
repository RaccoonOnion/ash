import 'dart:math' as math;
import 'package:flutter/material.dart';
import '../theme/ash_theme.dart';

/// CustomPaint widget rendering the Ash logo mark:
/// a centre dot, triangle outline, and crossbar line.
class AshMark extends StatelessWidget {
  const AshMark({
    super.key,
    this.size = 28,
    this.color,
    this.glow = false,
  });

  /// Logical size of the mark (width == height).
  final double size;

  /// Override colour. Defaults to [AshColors.text].
  final Color? color;

  /// When true, draws a blurred circle behind the centre dot.
  final bool glow;

  @override
  Widget build(BuildContext context) {
    final c = ash(context);
    final paintColor = color ?? c.text;

    return CustomPaint(
      size: Size(size, size),
      painter: _AshMarkPainter(paintColor, glow, c.accentGlow),
    );
  }
}

class _AshMarkPainter extends CustomPainter {
  _AshMarkPainter(this.color, this.glow, this.glowColor);

  final Color color;
  final bool glow;
  final Color glowColor;

  @override
  void paint(Canvas canvas, Size size) {
    final cx = size.width / 2;
    final cy = size.height / 2;
    final unit = size.width / 28; // design at 28dp

    final strokePaint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5 * unit
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;

    final fillPaint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;

    // Optional glow behind centre dot
    if (glow) {
      final glowPaint = Paint()
        ..color = glowColor
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 6);
      canvas.drawCircle(Offset(cx, cy), 5 * unit, glowPaint);
    }

    // Triangle outline — equilateral, pointing up
    final triRadius = 11 * unit;
    final triPath = Path();
    for (var i = 0; i < 3; i++) {
      final angle = -90.0 + i * 120.0;
      final rad = angle * 3.14159265 / 180.0;
      final x = cx + triRadius * math.cos(rad);
      final y = cy + triRadius * math.sin(rad);
      if (i == 0) {
        triPath.moveTo(x, y);
      } else {
        triPath.lineTo(x, y);
      }
    }
    triPath.close();
    canvas.drawPath(triPath, strokePaint);

    // Crossbar — horizontal line across the middle of the triangle
    final barY = cy + 2 * unit;
    final barHalf = triRadius * 0.55;
    canvas.drawLine(
      Offset(cx - barHalf, barY),
      Offset(cx + barHalf, barY),
      strokePaint,
    );

    // Centre dot (filled circle)
    canvas.drawCircle(Offset(cx, cy), 2.5 * unit, fillPaint);
  }

  @override
  bool shouldRepaint(covariant _AshMarkPainter old) =>
      color != old.color || glow != old.glow;
}
