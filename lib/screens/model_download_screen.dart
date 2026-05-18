import 'dart:math' as math;
import 'package:flutter/material.dart';
import '../theme/ash_theme.dart';
import '../widgets/glass.dart';
import '../widgets/buttons.dart';
import 'model_pick_screen.dart' show ModelInfo;

/// Download progress screen with animated ring.
class ModelDownloadScreen extends StatefulWidget {
  const ModelDownloadScreen({
    super.key,
    required this.model,
    required this.progress,
    required this.etaMin,
    required this.onContinue,
    this.onBack,
  });

  final ModelInfo model;
  final double progress; // 0.0 – 1.0
  final int etaMin;
  final VoidCallback onContinue;
  /// Leave the download screen without cancelling. The install keeps
  /// running in the background — user can re-enter via the bottom banner.
  /// Null for the onboarding flow (we don't want users skipping the first
  /// download), non-null when entered from Settings.
  final VoidCallback? onBack;

  @override
  State<ModelDownloadScreen> createState() => _ModelDownloadScreenState();
}

class _ModelDownloadScreenState extends State<ModelDownloadScreen>
    with SingleTickerProviderStateMixin {
  late final AnimationController _glowController;

  @override
  void initState() {
    super.initState();
    _glowController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1800),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _glowController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final c = ash(context);
    final pct = (widget.progress * 100).round();

    return Scaffold(
      backgroundColor: c.bg,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24),
          child: Column(
            children: [
              // Back chevron (only when re-entering from Settings — the
              // first-run onboarding download has nowhere to back out to).
              // Download keeps running in the background; user re-enters via
              // the bottom banner.
              if (widget.onBack != null)
                Align(
                  alignment: Alignment.centerLeft,
                  child: IconBtn(
                    icon: Icons.arrow_back,
                    onTap: widget.onBack!,
                  ),
                )
              else
                const SizedBox(height: 24),

              const SizedBox(height: 24),

              // ── Label ────────────────────────────────────
              Text(
                'Downloading model',
                style: TextStyle(
                  color: c.textDim,
                  fontSize: 13,
                  fontWeight: FontWeight.w500,
                  letterSpacing: 0.4,
                ),
              ),
              const SizedBox(height: 8),

              // ── Title ────────────────────────────────────
              Text(
                '${widget.model.name} ${widget.model.param}',
                style: TextStyle(
                  color: c.text,
                  fontSize: 22,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                widget.model.vendor,
                style: TextStyle(
                  color: c.textMuted,
                  fontSize: 14,
                ),
              ),

              // ── Progress ring ────────────────────────────
              const Spacer(flex: 1),
              SizedBox(
                width: 220,
                height: 220,
                child: Stack(
                  alignment: Alignment.center,
                  children: [
                    // Pulsing ember glow behind ring
                    AnimatedBuilder(
                      animation: _glowController,
                      builder: (context, _) {
                        final glowScale = 0.7 + _glowController.value * 0.3;
                        return Transform.scale(
                          scale: glowScale,
                          child: Container(
                            width: 160,
                            height: 160,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              boxShadow: [
                                BoxShadow(
                                  color: c.accentGlow.withValues(
                                      alpha: 0.5 * _glowController.value),
                                  blurRadius: 40,
                                  spreadRadius: 10,
                                ),
                              ],
                            ),
                          ),
                        );
                      },
                    ),

                    // Ring + text
                    CustomPaint(
                      size: const Size(220, 220),
                      painter: _ProgressRingPainter(
                        progress: widget.progress,
                        trackColor: c.surfaceStrong,
                        fillColor: c.accent,
                        strokeWidth: 6,
                      ),
                    ),

                    // Centre text
                    Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text(
                          pct >= 100 ? 'Ready' : '$pct%',
                          style: TextStyle(
                            color: c.text,
                            fontSize: 36,
                            fontWeight: FontWeight.w600,
                            letterSpacing: -0.5,
                          ),
                        ),
                        if (pct < 100) ...[
                          const SizedBox(height: 2),
                          Text(
                            widget.etaMin > 0
                                ? '${widget.etaMin} min left'
                                : 'Calculating…',
                            style: TextStyle(
                              color: c.textDim,
                              fontSize: 13,
                            ),
                          ),
                        ],
                      ],
                    ),
                  ],
                ),
              ),
              const Spacer(flex: 1),

              // ── Info card ────────────────────────────────
              Glass(
                radius: 18,
                padding: const EdgeInsets.all(16),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(Icons.info_outline, size: 18, color: c.textDim),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        'You can leave the app \u2014 Ash keeps downloading in the background.',
                        style: TextStyle(
                          color: c.textMuted,
                          fontSize: 13.5,
                          height: 1.4,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 20),

              // ── CTA ──────────────────────────────────────
              PrimaryBtn(
                label: 'Continue setting up',
                onTap: widget.onContinue,
              ),
              const SizedBox(height: 16),
            ],
          ),
        ),
      ),
    );
  }
}

// ─── Progress ring painter ────────────────────────────────────────────────────

class _ProgressRingPainter extends CustomPainter {
  _ProgressRingPainter({
    required this.progress,
    required this.trackColor,
    required this.fillColor,
    required this.strokeWidth,
  });

  final double progress;
  final Color trackColor;
  final Color fillColor;
  final double strokeWidth;

  @override
  void paint(Canvas canvas, Size size) {
    final cx = size.width / 2;
    final cy = size.height / 2;
    final radius = (size.width - strokeWidth) / 2;

    // Track
    final trackPaint = Paint()
      ..color = trackColor
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..strokeCap = StrokeCap.round;
    canvas.drawCircle(Offset(cx, cy), radius, trackPaint);

    // Fill arc
    final fillPaint = Paint()
      ..color = fillColor
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..strokeCap = StrokeCap.round;

    const startAngle = -math.pi / 2;
    final sweepAngle = 2 * math.pi * progress.clamp(0.0, 1.0);

    if (progress > 0) {
      canvas.drawArc(
        Rect.fromCircle(center: Offset(cx, cy), radius: radius),
        startAngle,
        sweepAngle,
        false,
        fillPaint,
      );
    }
  }

  @override
  bool shouldRepaint(covariant _ProgressRingPainter old) =>
      progress != old.progress;
}
