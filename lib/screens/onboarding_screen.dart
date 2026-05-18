import 'dart:math' as math;
import 'package:flutter/material.dart';
import '../theme/ash_theme.dart';
import '../widgets/glass.dart';
import '../widgets/buttons.dart';

/// 3-slide onboarding carousel.
class OnboardingScreen extends StatefulWidget {
  const OnboardingScreen({super.key, required this.onDone});

  final VoidCallback onDone;

  @override
  State<OnboardingScreen> createState() => _OnboardingScreenState();
}

class _OnboardingScreenState extends State<OnboardingScreen>
    with TickerProviderStateMixin {
  int _step = 0;
  late final AnimationController _emberController;

  static const _slides = [
    (
      title: 'Knowledge\nyou keep.',
      body:
          'Ash stores survival, medical, and repair knowledge right on your device. No signal required.',
    ),
    (
      title: 'Three ways\nto ask.',
      body:
          'Type a question, talk out loud, or point your camera at something — Ash understands all three.',
    ),
    (
      title: 'Pack what\nyou need.',
      body:
          'Choose knowledge packs tailored to your environment. Download once, keep forever.',
    ),
  ];

  @override
  void initState() {
    super.initState();
    _emberController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 3),
    )..repeat();
  }

  @override
  void dispose() {
    _emberController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final c = ash(context);
    final slide = _slides[_step];

    return Scaffold(
      backgroundColor: c.bg,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // ── Skip button ──────────────────────────────
              Align(
                alignment: Alignment.topRight,
                child: Padding(
                  padding: const EdgeInsets.only(top: 8),
                  child: GestureDetector(
                    onTap: widget.onDone,
                    child: Text(
                      'Skip',
                      style: TextStyle(
                        color: c.textDim,
                        fontSize: 15,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 16),

              // ── Visual area ──────────────────────────────
              Expanded(
                flex: 1,
                child: Center(
                  child: _buildVisual(c),
                ),
              ),

              // ── Page indicator dots ──────────────────────
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 20),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: List.generate(3, (i) {
                    final active = i == _step;
                    return AnimatedContainer(
                      duration: const Duration(milliseconds: 300),
                      margin: const EdgeInsets.symmetric(horizontal: 4),
                      width: active ? 22 : 6,
                      height: 6,
                      decoration: BoxDecoration(
                        color: active ? c.accent : c.border,
                        borderRadius: BorderRadius.circular(3),
                      ),
                    );
                  }),
                ),
              ),

              // ── Title ────────────────────────────────────
              Text(
                slide.title,
                style: TextStyle(
                  color: c.text,
                  fontSize: 46,
                  fontWeight: FontWeight.w400,
                  letterSpacing: -1.2,
                  height: 1.1,
                ),
              ),
              const SizedBox(height: 14),

              // ── Body ─────────────────────────────────────
              Text(
                slide.body,
                style: TextStyle(
                  color: c.textMuted,
                  fontSize: 15.5,
                  height: 1.45,
                ),
              ),
              const SizedBox(height: 28),

              // ── CTA button ───────────────────────────────
              PrimaryBtn(
                label: _step == 2 ? 'Get started' : 'Continue',
                onTap: () {
                  if (_step < 2) {
                    setState(() => _step++);
                  } else {
                    widget.onDone();
                  }
                },
              ),
              const SizedBox(height: 16),
            ],
          ),
        ),
      ),
    );
  }

  /// Builds the visual illustration for the current slide.
  Widget _buildVisual(AshColors c) {
    switch (_step) {
      case 0:
        return _EmberCore(controller: _emberController, colors: c);
      case 1:
        return _ThreeWaysVisual(colors: c);
      case 2:
        return _PackGridVisual(colors: c);
      default:
        return const SizedBox.shrink();
    }
  }
}

// ─── Slide 0: Animated ember core ────────────────────────────────────────────

class _EmberCore extends StatelessWidget {
  const _EmberCore({required this.controller, required this.colors});
  final AnimationController controller;
  final AshColors colors;

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: controller,
      builder: (context, _) {
        final t = controller.value;
        return SizedBox(
          width: 200,
          height: 200,
          child: CustomPaint(
            painter: _EmberPainter(colors: colors, t: t),
          ),
        );
      },
    );
  }
}

class _EmberPainter extends CustomPainter {
  _EmberPainter({required this.colors, required this.t});
  final AshColors colors;
  final double t;

  @override
  void paint(Canvas canvas, Size size) {
    final cx = size.width / 2;
    final cy = size.height / 2;

    // Outer glow ring
    final glowPaint = Paint()
      ..color = colors.accentGlow
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 20);
    canvas.drawCircle(
      Offset(cx, cy),
      70 + math.sin(t * 2 * math.pi) * 5,
      glowPaint,
    );

    // Concentric circles
    final rings = [80.0, 56.0, 32.0];
    for (final r in rings) {
      final animatedR = r + math.sin(t * 2 * math.pi + r) * 3;
      final paint = Paint()
        ..color = colors.accent.withValues(
          alpha: 0.15 + (0.1 * (80 - r) / 80),
        )
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.2;
      canvas.drawCircle(Offset(cx, cy), animatedR, paint);
    }

    // Inner filled core
    final corePaint = Paint()
      ..shader = RadialGradient(
        colors: [colors.accent, colors.accent.withValues(alpha: 0.6)],
      ).createShader(Rect.fromCircle(center: Offset(cx, cy), radius: 20));
    canvas.drawCircle(
      Offset(cx, cy),
      18 + math.sin(t * 2 * math.pi) * 2,
      corePaint,
    );

    // Center bright dot
    canvas.drawCircle(
      Offset(cx, cy),
      6,
      Paint()..color = Colors.white.withValues(alpha: 0.8),
    );
  }

  @override
  bool shouldRepaint(covariant _EmberPainter old) => t != old.t;
}

// ─── Slide 1: Three glass cards ──────────────────────────────────────────────

class _ThreeWaysVisual extends StatelessWidget {
  const _ThreeWaysVisual({required this.colors});
  final AshColors colors;

  @override
  Widget build(BuildContext context) {
    final items = [
      (Icons.keyboard, 'Type'),
      (Icons.mic_none_outlined, 'Talk'),
      (Icons.camera_alt_outlined, 'Camera'),
    ];

    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: items.map((item) {
        return Padding(
          padding: const EdgeInsets.symmetric(horizontal: 8),
          child: Glass(
            radius: 20,
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 24),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(item.$1, size: 28, color: colors.accent),
                const SizedBox(height: 10),
                Text(
                  item.$2,
                  style: TextStyle(
                    color: colors.text,
                    fontSize: 13,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ],
            ),
          ),
        );
      }).toList(),
    );
  }
}

// ─── Slide 2: Pack grid ──────────────────────────────────────────────────────

class _PackGridVisual extends StatelessWidget {
  const _PackGridVisual({required this.colors});
  final AshColors colors;

  @override
  Widget build(BuildContext context) {
    final packs = [
      ('Survival', Icons.terrain),
      ('Medical', Icons.medical_services_outlined),
      ('Repair', Icons.build_outlined),
      ('Navigation', Icons.explore_outlined),
      ('Weather', Icons.cloud_outlined),
      ('Foraging', Icons.eco_outlined),
    ];

    return Wrap(
      spacing: 10,
      runSpacing: 10,
      alignment: WrapAlignment.center,
      children: packs.map((pack) {
        return Glass(
          radius: 16,
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(pack.$2, size: 22, color: colors.accent),
              const SizedBox(height: 6),
              Text(
                pack.$1,
                style: TextStyle(
                  color: colors.textMuted,
                  fontSize: 11,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ],
          ),
        );
      }).toList(),
    );
  }
}
