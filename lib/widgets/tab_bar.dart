import 'package:flutter/material.dart';
import '../theme/ash_theme.dart';
import '../models/app_state.dart';
import 'glass.dart';

/// Bottom navigation bar rendered as a glass pill.
class AshTabBar extends StatelessWidget {
  const AshTabBar({
    required this.activeTab,
    required this.onChanged,
    super.key,
  });

  final MainTab activeTab;
  final ValueChanged<MainTab> onChanged;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: Glass(
        radius: 32,
        strong: true,
        padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 4),
        child: Row(
          children: [
            _TabButton(
              icon: Icons.chat_bubble_outline,
              label: 'Chat',
              tab: MainTab.home,
              activeTab: activeTab,
              onChanged: onChanged,
            ),
            _TabButton(
              icon: Icons.menu_book_outlined,
              label: 'Knowledge',
              tab: MainTab.knowledge,
              activeTab: activeTab,
              onChanged: onChanged,
            ),
            _TabButton(
              icon: Icons.memory_outlined,
              label: 'Models',
              tab: MainTab.models,
              activeTab: activeTab,
              onChanged: onChanged,
            ),
            _TabButton(
              icon: Icons.person_outline,
              label: 'You',
              tab: MainTab.profile,
              activeTab: activeTab,
              onChanged: onChanged,
            ),
          ],
        ),
      ),
    );
  }
}

class _TabButton extends StatelessWidget {
  const _TabButton({
    required this.icon,
    required this.label,
    required this.tab,
    required this.activeTab,
    required this.onChanged,
  });

  final IconData icon;
  final String label;
  final MainTab tab;
  final MainTab activeTab;
  final ValueChanged<MainTab> onChanged;

  @override
  Widget build(BuildContext context) {
    final c = ash(context);
    final isActive = tab == activeTab;

    return Expanded(
      child: GestureDetector(
        onTap: () => onChanged(tab),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          curve: Curves.easeOut,
          padding: const EdgeInsets.symmetric(vertical: 10),
          decoration: BoxDecoration(
            color: isActive ? c.accentSoft : Colors.transparent,
            borderRadius: BorderRadius.circular(28),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                icon,
                size: 20,
                color: isActive ? c.accent : c.textMuted,
              ),
              if (isActive) ...[
                const SizedBox(height: 3),
                Text(
                  label,
                  style: TextStyle(
                    color: c.accent,
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                    height: 1,
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
