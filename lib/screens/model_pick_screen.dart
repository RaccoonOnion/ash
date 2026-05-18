import 'package:flutter/material.dart' hide Chip;
import '../theme/ash_theme.dart';
import '../widgets/glass.dart';
import '../widgets/buttons.dart';
import '../widgets/chips.dart';

// ─── ModelInfo ────────────────────────────────────────────────────────────────

class ModelInfo {
  final String id;
  final String vendor;
  final String name;
  final String param;
  final String size;
  final String blurb;
  final String badge;
  final double bytes;
  final int eta;
  final int speed;
  final int quality;

  const ModelInfo({
    required this.id,
    required this.vendor,
    required this.name,
    required this.param,
    required this.size,
    required this.blurb,
    required this.badge,
    required this.bytes,
    required this.eta,
    required this.speed,
    required this.quality,
  });
}

const seedModels = [
  ModelInfo(
    id: 'gemma-4-E2B-it',
    vendor: 'Google',
    name: 'Gemma 4',
    param: 'E2B',
    size: '1.4 GB',
    bytes: 1.4,
    blurb: 'Balanced quality and speed. Runs fully on-device with RAG.',
    eta: 12,
    speed: 4,
    quality: 4,
    badge: 'Recommended',
  ),
  ModelInfo(
    id: 'gemma-4-E4B-it',
    vendor: 'Google',
    name: 'Gemma 4',
    param: 'E4B',
    size: '3.7 GB',
    bytes: 3.7,
    // Tradeoff intentionally surfaced — users on smaller iPhones should
    // think twice before downloading. Speed/quality dots are subjective
    // but reflect E4B's larger MoE: better reasoning, slower TTFT.
    blurb: 'Higher quality reasoning. Best on iPhone 17 Pro+. Larger download.',
    eta: 30,
    speed: 2,
    quality: 5,
    badge: 'High quality',
  ),
];

// ─── ModelPickScreen ─────────────────────────────────────────────────────────

class ModelPickScreen extends StatefulWidget {
  const ModelPickScreen({super.key, required this.onSelect});

  final ValueChanged<ModelInfo> onSelect;

  @override
  State<ModelPickScreen> createState() => _ModelPickScreenState();
}

class _ModelPickScreenState extends State<ModelPickScreen> {
  int _selected = 0; // default: first (Recommended)

  @override
  Widget build(BuildContext context) {
    final c = ash(context);
    final model = seedModels[_selected];

    return Scaffold(
      backgroundColor: c.bg,
      body: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // ── Header ─────────────────────────────────────
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const SizedBox(height: 16),
                  Text(
                    'Step 4 of 4 \u00B7 Pick your brain',
                    style: TextStyle(
                      color: c.textDim,
                      fontSize: 13,
                      fontWeight: FontWeight.w500,
                      letterSpacing: 0.4,
                    ),
                  ),
                  const SizedBox(height: 10),
                  Text(
                    'Choose the model Ash will run.',
                    style: TextStyle(
                      color: c.text,
                      fontSize: 38,
                      fontWeight: FontWeight.w400,
                      letterSpacing: -0.8,
                      height: 1.15,
                    ),
                  ),
                  const SizedBox(height: 20),
                ],
              ),
            ),

            // ── Model list ─────────────────────────────────
            Expanded(
              child: ListView.separated(
                padding: const EdgeInsets.symmetric(horizontal: 24),
                itemCount: seedModels.length + 1, // +1 for info card
                separatorBuilder: (_, __) => const SizedBox(height: 12),
                itemBuilder: (context, index) {
                  if (index == seedModels.length) {
                    return _PrivacyInfoCard(colors: c);
                  }
                  return _ModelCard(
                    model: seedModels[index],
                    isSelected: index == _selected,
                    onTap: () => setState(() => _selected = index),
                  );
                },
              ),
            ),

            // ── CTA button ─────────────────────────────────
            Padding(
              padding: const EdgeInsets.fromLTRB(24, 12, 24, 16),
              child: PrimaryBtn(
                label:
                    'Download ${model.name} ${model.param} \u00B7 ${model.size}',
                onTap: () => widget.onSelect(model),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ─── Model card ───────────────────────────────────────────────────────────────

class _ModelCard extends StatelessWidget {
  const _ModelCard({
    required this.model,
    required this.isSelected,
    required this.onTap,
  });

  final ModelInfo model;
  final bool isSelected;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final c = ash(context);

    return GestureDetector(
      onTap: onTap,
      child: Glass(
        radius: 20,
        padding: const EdgeInsets.all(16),
        borderColor: isSelected ? c.accent : null,
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Radio circle
            Container(
              width: 22,
              height: 22,
              margin: const EdgeInsets.only(top: 2),
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                border: Border.all(
                  color: isSelected ? c.accent : c.border,
                  width: isSelected ? 6 : 1.5,
                ),
              ),
            ),
            const SizedBox(width: 14),

            // Content
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Top row: vendor / name / param + badge
                  Row(
                    children: [
                      Text(
                        '${model.vendor} ',
                        style: TextStyle(
                          color: c.textDim,
                          fontSize: 13,
                        ),
                      ),
                      Text(
                        '${model.name} ',
                        style: TextStyle(
                          color: c.text,
                          fontSize: 15,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      Text(
                        model.param,
                        style: TextStyle(
                          color: c.textMuted,
                          fontSize: 14,
                        ),
                      ),
                      if (model.badge.isNotEmpty) ...[
                        const SizedBox(width: 8),
                        Chip(label: model.badge),
                      ],
                    ],
                  ),
                  const SizedBox(height: 6),

                  // Blurb
                  Text(
                    model.blurb,
                    style: TextStyle(
                      color: c.textMuted,
                      fontSize: 13.5,
                      height: 1.35,
                    ),
                  ),
                  const SizedBox(height: 12),

                  // Speed / Quality bars
                  Row(
                    children: [
                      _BarIndicator(
                        label: 'Speed',
                        value: model.speed,
                        max: 5,
                        colors: c,
                      ),
                      const SizedBox(width: 20),
                      _BarIndicator(
                        label: 'Quality',
                        value: model.quality,
                        max: 5,
                        colors: c,
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ─── Bar indicator ────────────────────────────────────────────────────────────

class _BarIndicator extends StatelessWidget {
  const _BarIndicator({
    required this.label,
    required this.value,
    required this.max,
    required this.colors,
  });

  final String label;
  final int value;
  final int max;
  final AshColors colors;

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: TextStyle(
              color: colors.textDim,
              fontSize: 11,
              fontWeight: FontWeight.w500,
            ),
          ),
          const SizedBox(height: 5),
          Row(
            children: List.generate(max, (i) {
              final filled = i < value;
              return Expanded(
                child: Container(
                  height: 4,
                  margin: const EdgeInsets.only(right: 3),
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(2),
                    color: filled ? colors.accent : colors.surfaceStrong,
                  ),
                ),
              );
            }),
          ),
        ],
      ),
    );
  }
}

// ─── Privacy info card ────────────────────────────────────────────────────────

class _PrivacyInfoCard extends StatelessWidget {
  const _PrivacyInfoCard({required this.colors});
  final AshColors colors;

  @override
  Widget build(BuildContext context) {
    return Glass(
      radius: 18,
      padding: const EdgeInsets.all(16),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.lock_outline, size: 18, color: colors.textDim),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              'Your data never leaves this device. All inference runs locally.',
              style: TextStyle(
                color: colors.textMuted,
                fontSize: 13,
                height: 1.4,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
