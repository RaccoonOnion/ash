import 'package:flutter/material.dart';
import '../theme/ash_theme.dart';
import 'glass.dart';
import 'buttons.dart';

/// Bottom sheet modal for camera capture with viewfinder.
class CameraSheet extends StatefulWidget {
  const CameraSheet({
    required this.onClose,
    required this.onComplete,
    super.key,
  });

  final VoidCallback onClose;
  final ValueChanged<String> onComplete;

  @override
  State<CameraSheet> createState() => _CameraSheetState();
}

class _CameraSheetState extends State<CameraSheet> {
  bool _capturing = false;

  void _capture() {
    setState(() => _capturing = true);
    Future.delayed(const Duration(seconds: 2), () {
      if (mounted) {
        widget.onComplete('Identify this plant — is it safe?');
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final c = ash(context);

    return Glass(
      radius: 28,
      strong: true,
      padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 20),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Drag handle
          Container(
            width: 36,
            height: 4,
            decoration: BoxDecoration(
              color: c.textDim,
              borderRadius: BorderRadius.circular(2),
            ),
          ),
          const SizedBox(height: 20),

          // Viewfinder
          if (!_capturing)
            AspectRatio(
              aspectRatio: 3 / 4,
              child: Container(
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(16),
                  gradient: LinearGradient(
                    begin: Alignment.topCenter,
                    end: Alignment.bottomCenter,
                    colors: [
                      c.surfaceStrong,
                      c.bg,
                    ],
                  ),
                ),
                child: Stack(
                  children: [
                    // Corner markers
                    _Corner(c: c, alignment: Alignment.topLeft),
                    _Corner(c: c, alignment: Alignment.topRight),
                    _Corner(c: c, alignment: Alignment.bottomLeft),
                    _Corner(c: c, alignment: Alignment.bottomRight),

                    // Label
                    Center(
                      child: Text(
                        'Frame what you want Ash to see',
                        style: TextStyle(
                          color: c.textDim,
                          fontSize: 13,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ),
                  ],
                ),
              ),
            )
          else
            Container(
              height: 200,
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(16),
                color: c.surfaceStrong,
              ),
              child: Center(
                child: SizedBox(
                  width: 40,
                  height: 40,
                  child: CircularProgressIndicator(
                    strokeWidth: 3,
                    color: c.accent,
                  ),
                ),
              ),
            ),

          const SizedBox(height: 24),

          // Capture / Close row
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              IconBtn(
                icon: Icons.close,
                onTap: widget.onClose,
              ),
              if (!_capturing)
                GestureDetector(
                  onTap: _capture,
                  child: Container(
                    width: 64,
                    height: 64,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      border: Border.all(color: c.text, width: 3),
                    ),
                    child: Center(
                      child: Container(
                        width: 48,
                        height: 48,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: c.accent,
                        ),
                      ),
                    ),
                  ),
                )
              else
                const SizedBox(width: 64),
              // Spacer for alignment
              const SizedBox(width: 36),
            ],
          ),
        ],
      ),
    );
  }
}

/// Corner marker for the viewfinder frame.
class _Corner extends StatelessWidget {
  const _Corner({required this.c, required this.alignment});

  final AshColors c;
  final Alignment alignment;

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: alignment,
      child: CustomPaint(
        size: const Size(28, 28),
        painter: _CornerPainter(c.accent, alignment),
      ),
    );
  }
}

class _CornerPainter extends CustomPainter {
  _CornerPainter(this.color, this.alignment);

  final Color color;
  final Alignment alignment;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..strokeWidth = 2.5
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    final w = size.width;
    final h = size.height;
    final path = Path();

    final isLeft = alignment == Alignment.topLeft ||
        alignment == Alignment.bottomLeft;
    final isTop = alignment == Alignment.topLeft ||
        alignment == Alignment.topRight;

    if (isTop && isLeft) {
      path.moveTo(0, h);
      path.lineTo(0, 0);
      path.lineTo(w, 0);
    } else if (isTop && !isLeft) {
      path.moveTo(0, 0);
      path.lineTo(w, 0);
      path.lineTo(w, h);
    } else if (!isTop && isLeft) {
      path.moveTo(0, 0);
      path.lineTo(0, h);
      path.lineTo(w, h);
    } else {
      path.moveTo(w, 0);
      path.lineTo(w, h);
      path.lineTo(0, h);
    }

    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant _CornerPainter old) =>
      color != old.color || alignment != old.alignment;
}
