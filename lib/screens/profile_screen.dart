import 'package:flutter/material.dart';
import 'package:share_plus/share_plus.dart';
import '../services/llm_model.dart';
import '../theme/ash_theme.dart';
import '../widgets/glass.dart';
import '../widgets/buttons.dart';

/// User profile tab. The "Active model" row is the primary control: tap to
/// open a picker showing every [LlmModel] variant with install state, switch
/// between them, or download a new one. Other rows surface device + storage
/// info plus a small action set. No login = no account state to display.
class ProfileScreen extends StatelessWidget {
  const ProfileScreen({
    required this.onOpenSettings,
    required this.activeLlmModel,
    required this.installedLlmModels,
    required this.onSwitchLlmModel,
    required this.onInstallLlmModel,
    required this.onDeleteLlmModel,
    required this.storageBytesUsed,
    required this.packsInstalled,
    required this.chatsCount,
    super.key,
  });

  final VoidCallback onOpenSettings;
  final LlmModel activeLlmModel;
  final Set<LlmModel> installedLlmModels;
  final ValueChanged<LlmModel> onSwitchLlmModel;
  final ValueChanged<LlmModel> onInstallLlmModel;
  /// Trash-icon tap on an inactive installed variant.
  final ValueChanged<LlmModel> onDeleteLlmModel;
  /// Real on-disk usage in bytes (sum of installed models + active packs).
  final double storageBytesUsed;
  final int packsInstalled;
  final int chatsCount;

  @override
  Widget build(BuildContext context) {
    final c = ash(context);

    return SafeArea(
      child: ListView(
        padding: const EdgeInsets.fromLTRB(20, 16, 20, 40),
        children: [
          // Title row — settings icon replaces the logout-laden identity card.
          Row(
            children: [
              Expanded(
                child: Text(
                  'You',
                  style: TextStyle(
                    color: c.text,
                    fontSize: 36,
                    fontWeight: FontWeight.w400,
                  ),
                ),
              ),
              IconBtn(
                icon: Icons.settings,
                onTap: onOpenSettings,
                size: 38,
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            'Everything runs on this device. No account needed.',
            style: TextStyle(color: c.textMuted, fontSize: 13),
          ),
          const SizedBox(height: 20),

          // Stats row — real values, no fake account data.
          Row(
            children: [
              Expanded(child: _statCard(c, 'Packs', '$packsInstalled')),
              const SizedBox(width: 10),
              Expanded(child: _statCard(c, 'Conversations', '$chatsCount')),
              const SizedBox(width: 10),
              Expanded(
                  child: _statCard(c, 'On-device', _fmtBytes(storageBytesUsed))),
            ],
          ),
          const SizedBox(height: 24),

          // Model section — the headline control for this page.
          const _SectionHeader(label: 'Model'),
          const SizedBox(height: 8),
          Glass(
            radius: 18,
            padding: const EdgeInsets.symmetric(vertical: 4),
            child: Column(
              children: [
                _row(
                  c,
                  icon: Icons.memory,
                  label: 'Active model',
                  value: activeLlmModel.displayName,
                  chevron: true,
                  onTap: () => _openModelPicker(context),
                ),
                _divider(c),
                _row(
                  c,
                  icon: Icons.bolt,
                  label: 'Inference',
                  value: 'On-device',
                ),
                _divider(c),
                _row(
                  c,
                  icon: Icons.storage_outlined,
                  label: 'Storage used',
                  value: _fmtBytes(storageBytesUsed),
                ),
              ],
            ),
          ),
          const SizedBox(height: 24),

          // Single quick action — sharing. The Settings row is gone:
          // the gear icon top-right of this page (and the rebuilt Settings
          // surface) is now the only entry point, removing a redundant
          // navigation path.
          const _SectionHeader(label: 'Share'),
          const SizedBox(height: 8),
          Glass(
            radius: 18,
            padding: const EdgeInsets.symmetric(vertical: 4),
            child: _row(
              c,
              icon: Icons.share_outlined,
              label: 'Share Ash',
              onTap: () => _shareLink(context),
            ),
          ),
          const SizedBox(height: 32),

          // Version footer.
          Center(
            child: Text(
              'Ash 1.4.0 · Built for moments without signal',
              style: TextStyle(color: c.textDim, fontSize: 12),
            ),
          ),
        ],
      ),
    );
  }

  /// Bottom sheet listing every [LlmModel]. Tap installed → switch active.
  /// Tap uninstalled → host pushes into the download flow.
  void _openModelPicker(BuildContext context) async {
    final c = ash(context);
    final selected = await showModalBottomSheet<LlmModel>(
      context: context,
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      builder: (sheetContext) => SafeArea(
        top: false,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
          child: Glass(
            radius: 24,
            strong: true,
            padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 4),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  width: 36,
                  height: 4,
                  margin: const EdgeInsets.only(bottom: 12),
                  decoration: BoxDecoration(
                    color: c.textDim,
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  child: Align(
                    alignment: Alignment.centerLeft,
                    child: Text(
                      'Choose model',
                      style: TextStyle(
                        color: c.text,
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 8),
                for (final m in LlmModel.values)
                  _ModelOptionRow(
                    model: m,
                    isActive: m == activeLlmModel,
                    isInstalled: installedLlmModels.contains(m),
                    onTap: () => Navigator.of(sheetContext).pop(m),
                    onDelete: () async {
                      Navigator.of(sheetContext).pop();
                      await _confirmDelete(context, m);
                    },
                  ),
              ],
            ),
          ),
        ),
      ),
    );
    if (selected == null || selected == activeLlmModel) return;
    if (installedLlmModels.contains(selected)) {
      onSwitchLlmModel(selected);
    } else {
      onInstallLlmModel(selected);
    }
  }

  /// Confirm + delegate model deletion. Reachable for any installed variant
  /// — the host handles the "deleting the active one" case by switching to
  /// another installed model first, or routing back to ModelPick if there's
  /// nothing left to fall back to.
  Future<void> _confirmDelete(BuildContext context, LlmModel m) async {
    final isActive = m == activeLlmModel;
    final hasFallback = installedLlmModels.any((other) => other != m);
    final isLast = isActive && !hasFallback;
    final body = isLast
        ? 'This is the only model on the device. Deleting it will take you back to the model picker so you can choose a different one.'
        : isActive
            ? 'This is the active model. Ash will automatically switch to another installed model after deleting.'
            : 'Frees ${m.sizeGB.toStringAsFixed(1)} GB on this device. You can reinstall later from the model picker.';
    final confirm = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text('Delete ${m.displayName}?'),
        content: Text(body),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(true),
            child: const Text(
              'Delete',
              style: TextStyle(color: Colors.redAccent),
            ),
          ),
        ],
      ),
    );
    if (confirm == true) onDeleteLlmModel(m);
  }

  /// Open the native iOS share sheet (Messages / Mail / AirDrop / etc.)
  /// with a link to the public repo. Subject is a hint Mail picks up.
  void _shareLink(BuildContext context) async {
    final params = ShareParams(
      text: 'Check out Ash — an offline survival assistant.\n'
          'https://github.com/RaccoonOnion/ash',
      subject: 'Ash — offline survival AI',
    );
    await SharePlus.instance.share(params);
  }

  Widget _statCard(AshColors c, String label, String value) {
    return Glass(
      radius: 16,
      padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 12),
      child: Column(
        children: [
          Text(
            value,
            style: TextStyle(
              color: c.text,
              fontSize: 18,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            label,
            style: TextStyle(
              color: c.textDim,
              fontSize: 11,
              fontWeight: FontWeight.w600,
              letterSpacing: 0.3,
            ),
          ),
        ],
      ),
    );
  }

  Widget _row(
    AshColors c, {
    required IconData icon,
    required String label,
    String? value,
    Color? textColor,
    VoidCallback? onTap,
    bool chevron = false,
  }) {
    final content = Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 13),
      child: Row(
        children: [
          Icon(icon, size: 18, color: textColor ?? c.textDim),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              label,
              style: TextStyle(
                color: textColor ?? c.text,
                fontSize: 15,
              ),
            ),
          ),
          if (value != null)
            Text(
              value,
              style: TextStyle(color: c.textMuted, fontSize: 13),
            ),
          if (chevron) ...[
            const SizedBox(width: 4),
            Icon(Icons.chevron_right, size: 20, color: c.textDim),
          ],
        ],
      ),
    );
    if (onTap == null) return content;
    return InkWell(onTap: onTap, child: content);
  }

  Widget _divider(AshColors c) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: Divider(height: 1, color: c.border),
    );
  }

  static String _fmtBytes(double bytesGB) {
    if (bytesGB >= 1.0) return '${bytesGB.toStringAsFixed(1)} GB';
    return '${(bytesGB * 1024).round()} MB';
  }
}

/// One row in the model-picker sheet. Same shape as the one used in
/// SettingsScreen — kept local to this file rather than pulled into a
/// shared widget because the two screens render almost identical chrome
/// and unifying them now would just add an indirection.
class _ModelOptionRow extends StatelessWidget {
  const _ModelOptionRow({
    required this.model,
    required this.isActive,
    required this.isInstalled,
    required this.onTap,
    this.onDelete,
  });

  final LlmModel model;
  final bool isActive;
  final bool isInstalled;
  final VoidCallback onTap;
  /// Rendered for any installed variant — including the active one. The
  /// host's delete handler auto-switches to another installed model before
  /// removing the active variant's file (or routes back to ModelPick if
  /// the deleted one was the last on disk).
  final VoidCallback? onDelete;

  @override
  Widget build(BuildContext context) {
    final c = ash(context);
    final (badgeLabel, badgeColor) = isActive
        ? ('Active', c.accent)
        : isInstalled
            ? ('Installed', c.text)
            : ('Download', c.textDim);
    return InkWell(
      onTap: onTap,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  Row(
                    children: [
                      Text(
                        model.displayName,
                        style: TextStyle(
                          color: c.text,
                          fontSize: 15,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      const SizedBox(width: 8),
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 6, vertical: 2),
                        decoration: BoxDecoration(
                          color: badgeColor.withValues(alpha: 0.15),
                          borderRadius: BorderRadius.circular(4),
                          border: Border.all(
                              color: badgeColor.withValues(alpha: 0.4),
                              width: 0.5),
                        ),
                        child: Text(
                          badgeLabel,
                          style: TextStyle(
                            color: badgeColor,
                            fontSize: 10,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 4),
                  Text(
                    '${model.sizeGB.toStringAsFixed(1)} GB · ${model.blurb}',
                    style: TextStyle(color: c.textDim, fontSize: 12),
                  ),
                ],
              ),
            ),
            if (isInstalled && onDelete != null)
              IconButton(
                icon: Icon(Icons.delete_outline, size: 20, color: c.textMuted),
                onPressed: onDelete,
                tooltip: isActive
                    ? 'Delete (auto-switches first)'
                    : 'Delete download',
                visualDensity: VisualDensity.compact,
              )
            else if (!isInstalled)
              Icon(Icons.download_outlined, size: 18, color: c.textMuted),
          ],
        ),
      ),
    );
  }
}

class _SectionHeader extends StatelessWidget {
  const _SectionHeader({required this.label});
  final String label;

  @override
  Widget build(BuildContext context) {
    final c = ash(context);
    return Text(
      label.toUpperCase(),
      style: TextStyle(
        color: c.textDim,
        fontSize: 11,
        fontWeight: FontWeight.w600,
        letterSpacing: 0.8,
      ),
    );
  }
}
