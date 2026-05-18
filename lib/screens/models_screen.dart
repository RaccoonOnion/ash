import 'package:flutter/material.dart';

import '../services/llm_model.dart';
import '../services/model_download_state.dart';
import '../theme/ash_theme.dart';
import '../widgets/glass.dart';

/// Centralised model-weights management surface.
///
/// One row per [LlmModel] variant. Each row owns the variant's full
/// lifecycle UI:
///   - idle / failed       → [Download] (or [Retry])
///   - downloading         → progress bar + Pause + Cancel
///   - paused              → frozen bar + Resume + Cancel
///   - installed inactive  → Switch (radio-style) + Delete
///   - installed active    → Active badge + Delete (host auto-switches)
///
/// Concurrent downloads are allowed — the host keeps per-variant
/// state/cancel-tokens so multiple bars run independently. Tapping a
/// Download row that's already in flight is a no-op.
class ModelsScreen extends StatelessWidget {
  const ModelsScreen({
    super.key,
    required this.activeLlmModel,
    required this.installedLlmModels,
    required this.downloads,
    required this.onDownload,
    required this.onCancel,
    required this.onSwitch,
    required this.onDelete,
  });

  final LlmModel activeLlmModel;
  final Set<LlmModel> installedLlmModels;

  /// Per-variant download lifecycle. Missing keys are treated as
  /// [DownloadState.idle].
  final Map<LlmModel, ModelDownloadInfo> downloads;

  final ValueChanged<LlmModel> onDownload;
  final ValueChanged<LlmModel> onCancel;
  final ValueChanged<LlmModel> onSwitch;
  /// Confirmed delete — host shows the confirmation, so this fires only
  /// when the user has acknowledged the (possibly-auto-switching) action.
  final ValueChanged<LlmModel> onDelete;

  @override
  Widget build(BuildContext context) {
    final c = ash(context);
    return SafeArea(
      bottom: false,
      child: ListView(
        padding: const EdgeInsets.fromLTRB(20, 16, 20, 32),
        children: [
          Text(
            'Models',
            style: TextStyle(
              color: c.text,
              fontSize: 28,
              fontWeight: FontWeight.w700,
              letterSpacing: -0.5,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            'Download, switch, and manage on-device Gemma variants. '
            'Multiple downloads can run at once.',
            style: TextStyle(
              color: c.textMuted,
              fontSize: 14,
              height: 1.4,
            ),
          ),
          const SizedBox(height: 20),
          for (final m in LlmModel.values) ...[
            _ModelRow(
              model: m,
              isActive: m == activeLlmModel,
              info: downloads[m] ?? const ModelDownloadInfo(),
              isInstalledOnDisk: installedLlmModels.contains(m),
              onDownload: () => onDownload(m),
              onCancel: () => _confirmCancel(context, m),
              onSwitch: () => onSwitch(m),
              onDelete: () => _confirmDelete(context, m),
            ),
            const SizedBox(height: 12),
          ],
        ],
      ),
    );
  }

  Future<void> _confirmDelete(BuildContext context, LlmModel m) async {
    final isActive = m == activeLlmModel;
    final hasFallback = installedLlmModels.any((other) => other != m);
    final isLast = isActive && !hasFallback;
    final body = isLast
        ? 'This is the only model on the device. Deleting it will take you back to the model picker so you can choose a different one.'
        : isActive
            ? 'This is the active model. Ash will automatically switch to another installed model after deleting.'
            : 'Frees ${m.sizeGB.toStringAsFixed(1)} GB on this device. You can reinstall later.';
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
    if (confirm == true) onDelete(m);
  }

  Future<void> _confirmCancel(BuildContext context, LlmModel m) async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text('Cancel ${m.displayName} download?'),
        content: const Text(
          'Cancelling discards the partial file. You can re-download later.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: const Text('Keep downloading'),
          ),
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(true),
            child: const Text(
              'Cancel download',
              style: TextStyle(color: Colors.redAccent),
            ),
          ),
        ],
      ),
    );
    if (confirm == true) onCancel(m);
  }
}

class _ModelRow extends StatelessWidget {
  const _ModelRow({
    required this.model,
    required this.isActive,
    required this.info,
    required this.isInstalledOnDisk,
    required this.onDownload,
    required this.onCancel,
    required this.onSwitch,
    required this.onDelete,
  });

  final LlmModel model;
  final bool isActive;
  final ModelDownloadInfo info;
  final bool isInstalledOnDisk;
  final VoidCallback onDownload;
  final VoidCallback onCancel;
  final VoidCallback onSwitch;
  final VoidCallback onDelete;

  @override
  Widget build(BuildContext context) {
    final c = ash(context);
    final state = info.state;
    // "On disk" = either we tracked an install via the state machine or
    // the SDK confirmed the file at launch (the legacy installed set).
    // Bias toward the SDK truth when the two disagree — the state map
    // might be stale across restarts.
    final installed = isInstalledOnDisk || state == DownloadState.installed;

    return Glass(
      radius: 18,
      padding: const EdgeInsets.fromLTRB(16, 14, 12, 14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Text(
                          model.displayName,
                          style: TextStyle(
                            color: c.text,
                            fontSize: 16,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                        const SizedBox(width: 8),
                        _StateBadge(
                          state: state,
                          isActive: isActive,
                          installed: installed,
                        ),
                      ],
                    ),
                    const SizedBox(height: 4),
                    Text(
                      '${model.sizeGB.toStringAsFixed(1)} GB · ${model.blurb}',
                      style: TextStyle(
                        color: c.textDim,
                        fontSize: 12.5,
                        height: 1.35,
                      ),
                    ),
                  ],
                ),
              ),
              _trailingAction(c, installed),
            ],
          ),
          if (state == DownloadState.downloading) ...[
            const SizedBox(height: 12),
            _ProgressStrip(info: info),
            const SizedBox(height: 10),
            Row(
              children: [
                _GhostBtn(
                  label: 'Cancel',
                  icon: Icons.close_rounded,
                  onTap: onCancel,
                ),
              ],
            ),
          ],
          if (state == DownloadState.failed && info.errorMessage != null) ...[
            const SizedBox(height: 10),
            Text(
              info.errorMessage!,
              style: TextStyle(color: Colors.redAccent, fontSize: 12),
            ),
            const SizedBox(height: 8),
            _GhostBtn(label: 'Retry download', onTap: onDownload),
          ],
        ],
      ),
    );
  }

  Widget _trailingAction(AshColors c, bool installed) {
    final state = info.state;
    // While downloading, the Cancel button lives below the progress
    // strip — no trailing button so the header row stays compact.
    if (state == DownloadState.downloading) {
      return const SizedBox.shrink();
    }
    if (installed) {
      // Installed-but-inactive → "Switch" (mini button) + delete icon.
      // Installed-and-active → just the delete icon (no switch needed).
      return Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (!isActive)
            _GhostBtn(label: 'Switch', onTap: onSwitch),
          IconButton(
            icon: Icon(Icons.delete_outline, size: 20, color: c.textMuted),
            onPressed: onDelete,
            tooltip:
                isActive ? 'Delete (auto-switches first)' : 'Delete download',
            visualDensity: VisualDensity.compact,
          ),
        ],
      );
    }
    // Not on disk, not in flight → Download button.
    return _PrimaryMiniBtn(
      label: state == DownloadState.failed ? 'Retry' : 'Download',
      icon: Icons.download_outlined,
      onTap: onDownload,
    );
  }
}

class _StateBadge extends StatelessWidget {
  const _StateBadge({
    required this.state,
    required this.isActive,
    required this.installed,
  });

  final DownloadState state;
  final bool isActive;
  final bool installed;

  @override
  Widget build(BuildContext context) {
    final c = ash(context);
    final (label, color) = isActive && installed
        ? ('Active', c.accent)
        : switch (state) {
            DownloadState.downloading => ('Downloading', c.accent),
            DownloadState.installed => ('Installed', c.text),
            DownloadState.failed => ('Failed', Colors.redAccent),
            DownloadState.idle =>
              installed ? ('Installed', c.text) : ('Not installed', c.textDim),
          };
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(4),
        border: Border.all(color: color.withValues(alpha: 0.4), width: 0.5),
      ),
      child: Text(
        label,
        style: TextStyle(
          color: color,
          fontSize: 10,
          fontWeight: FontWeight.w600,
          letterSpacing: 0.2,
        ),
      ),
    );
  }
}

class _ProgressStrip extends StatelessWidget {
  const _ProgressStrip({required this.info});
  final ModelDownloadInfo info;

  @override
  Widget build(BuildContext context) {
    final c = ash(context);
    final pct = (info.progress * 100).round();
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: LinearProgressIndicator(
            value: info.progress.clamp(0.0, 1.0),
            minHeight: 6,
            backgroundColor: c.surfaceStrong,
            valueColor: AlwaysStoppedAnimation<Color>(c.accent),
          ),
        ),
        const SizedBox(height: 6),
        Text(
          // etaMin == 0 means the estimator hasn't seen enough real
          // progress yet to commit to a number — show "Calculating…"
          // rather than a misleading baseline (or worse, a blank). Once
          // the first real measurement lands (typically within 1-2s of
          // download activity) the number takes over.
          info.etaMin > 0
              ? '$pct% · ~${info.etaMin} min left'
              : '$pct% · Calculating…',
          style: TextStyle(color: c.textDim, fontSize: 11.5),
        ),
      ],
    );
  }
}

class _GhostBtn extends StatelessWidget {
  const _GhostBtn({
    required this.label,
    required this.onTap,
    this.icon,
  });
  final String label;
  final VoidCallback onTap;
  final IconData? icon;

  @override
  Widget build(BuildContext context) {
    final c = ash(context);
    final fg = c.text;
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(12),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 7),
        decoration: BoxDecoration(
          color: c.surface,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: c.border, width: 0.6),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (icon != null) ...[
              Icon(icon, size: 14, color: fg),
              const SizedBox(width: 4),
            ],
            Text(
              label,
              style: TextStyle(
                color: fg,
                fontSize: 12.5,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _PrimaryMiniBtn extends StatelessWidget {
  const _PrimaryMiniBtn({
    required this.label,
    required this.onTap,
    required this.icon,
  });
  final String label;
  final VoidCallback onTap;
  final IconData icon;

  @override
  Widget build(BuildContext context) {
    final c = ash(context);
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(12),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: c.accent,
          borderRadius: BorderRadius.circular(12),
          boxShadow: [
            BoxShadow(
              color: c.accentGlow,
              blurRadius: 14,
              offset: const Offset(0, 0),
            ),
          ],
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 14, color: Colors.white),
            const SizedBox(width: 4),
            Text(
              label,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 13,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
