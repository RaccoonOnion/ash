import 'package:flutter/material.dart';
import '../theme/ash_theme.dart';

/// Small status pill showing a pulsing green dot and "On-device" text.
class OnDeviceBadge extends StatefulWidget {
  const OnDeviceBadge({super.key});

  @override
  State<OnDeviceBadge> createState() => _OnDeviceBadgeState();
}

class _OnDeviceBadgeState extends State<OnDeviceBadge>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
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

    return Container(
      height: 20,
      padding: const EdgeInsets.symmetric(horizontal: 8),
      decoration: BoxDecoration(
        color: c.surface,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: c.border, width: 0.5),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          AnimatedBuilder(
            animation: _controller,
            builder: (context, _) {
              final scale = 0.75 + 0.25 * _controller.value;
              return Transform.scale(
                scale: scale,
                child: Container(
                  width: 6,
                  height: 6,
                  decoration: BoxDecoration(
                    color: c.success,
                    shape: BoxShape.circle,
                    boxShadow: [
                      BoxShadow(
                        color: c.success.withValues(alpha: 0.6),
                        blurRadius: 4,
                      ),
                    ],
                  ),
                ),
              );
            },
          ),
          const SizedBox(width: 5),
          Text(
            'ON-DEVICE',
            style: TextStyle(
              color: c.textMuted,
              fontSize: 10.5,
              fontWeight: FontWeight.w600,
              letterSpacing: 0.5,
            ),
          ),
        ],
      ),
    );
  }
}
