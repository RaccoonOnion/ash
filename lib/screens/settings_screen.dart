import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import '../services/inference_service.dart';
import '../services/llm_model.dart';
import '../services/voice_service.dart';
import '../theme/ash_theme.dart';
import '../widgets/glass.dart';
import '../widgets/buttons.dart';
import '../widgets/chips.dart';

/// Full settings page (pushed on top, not a tab). Sampling/RAG knobs are
/// per-conversation and live in the chat header's tune icon, NOT here. The
/// only inference-related control on this page is the accelerator
/// (GPU/CPU) — that's engine-level and shared across chats.
class SettingsScreen extends StatefulWidget {
  const SettingsScreen({
    required this.onBack,
    required this.accelerator,
    required this.onAcceleratorChanged,
    required this.activeLlmModel,
    required this.installedLlmModels,
    required this.onSwitchLlmModel,
    required this.onInstallLlmModel,
    required this.onDeleteLlmModel,
    required this.onClearConversations,
    required this.voiceService,
    required this.speculativeDecoding,
    required this.onSpeculativeDecodingChanged,
    super.key,
  });

  final VoidCallback onBack;
  final AcceleratorChoice accelerator;
  final ValueChanged<AcceleratorChoice> onAcceleratorChanged;
  /// For the Voice section — list available iOS voices, preview, save the
  /// user's pick. Settings owns the picker UI; the service owns the
  /// persistence (NSUserDefaults under the hood).
  final VoiceService voiceService;
  /// Whether Multi-Token Prediction (speculative decoding) is enabled on
  /// the inference engine. Engine-level toggle — flipping it closes the
  /// engine and re-warms.
  final bool speculativeDecoding;
  /// Called when the user flips the MTP toggle. Host runs
  /// `service.setSpeculativeDecoding(...)` + re-warms; the engine-reload
  /// banner appears for ~5s while the new engine builds.
  final ValueChanged<bool> onSpeculativeDecodingChanged;
  /// Variant currently loaded into the engine.
  final LlmModel activeLlmModel;
  /// Variants present on disk (subset of [LlmModel.values]).
  final Set<LlmModel> installedLlmModels;
  /// Tap on an already-installed variant — host should call setActiveLlmModel
  /// and surface the engine-reload banner.
  final ValueChanged<LlmModel> onSwitchLlmModel;
  /// Tap on an uninstalled variant — host should drop into the download flow
  /// for that variant (will navigate to AppStage.downloading internally).
  final ValueChanged<LlmModel> onInstallLlmModel;
  /// Tap the trash icon on an installed-but-inactive variant. The host is
  /// responsible for confirming + calling service.deleteModel + refreshing
  /// the installed set.
  final ValueChanged<LlmModel> onDeleteLlmModel;
  /// Wipe all conversations. Host handles confirmation, state update.
  final VoidCallback onClearConversations;

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  @override
  Widget build(BuildContext context) {
    final c = ash(context);

    // Opaque background so the page underneath (Profile tab) doesn't bleed
    // through the Settings overlay.
    return ColoredBox(
      color: c.bg,
      child: SafeArea(
      child: Column(
        children: [
          // Header
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 8, 20, 0),
            child: Row(
              children: [
                IconBtn(
                  icon: Icons.arrow_back,
                  onTap: widget.onBack,
                ),
                const SizedBox(width: 8),
                Text(
                  'Settings',
                  style: TextStyle(
                    color: c.text,
                    fontSize: 28,
                    fontWeight: FontWeight.w400,
                  ),
                ),
              ],
            ),
          ),

          // Scrollable content
          Expanded(
            child: ListView(
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 40),
              children: [
                // ── Model ──────────────────────────────────────────────
                const SectionLabel(label: 'Model'),
                const SizedBox(height: 4),
                Glass(
                  radius: 18,
                  padding: const EdgeInsets.symmetric(vertical: 4),
                  child: Column(
                    children: [
                      _buildRow(
                        c,
                        label: 'Active model',
                        value: widget.activeLlmModel.displayName,
                        chevron: true,
                        onTap: _openModelPicker,
                      ),
                      _divider(c),
                      _buildAcceleratorRow(c),
                      _divider(c),
                      _buildSpeculativeRow(c),
                    ],
                  ),
                ),
                const SizedBox(height: 24),

                // Sampling controls (RAG / temperature / top-K / top-P /
                // max tokens) are per-conversation — accessed via the tune
                // icon in the chat header, not from this global page.

                // ── Voice ──────────────────────────────────────────────
                const SectionLabel(label: 'Voice'),
                const SizedBox(height: 4),
                Glass(
                  radius: 18,
                  padding: const EdgeInsets.symmetric(vertical: 4),
                  child: Column(
                    children: [
                      // Read-only voice display. Auto-picked at launch from
                      // installed Enhanced/Premium voices; the user no
                      // longer chooses. Default (robotic) voices are never
                      // selected — if no quality voice is installed this
                      // row shows "System default" and TTS falls back to
                      // whatever iOS picks.
                      _buildRow(
                        c,
                        label: 'Reply voice',
                        value: widget.voiceService.currentVoice?.name ??
                            'System default',
                        chevron: false,
                        onTap: null,
                      ),
                      _divider(c),
                      _buildSpeechRateRow(c),
                      _divider(c),
                      _buildListeningPatienceRow(c),
                    ],
                  ),
                ),
                const SizedBox(height: 24),

                // ── Storage ────────────────────────────────────────────
                const SectionLabel(label: 'Storage'),
                const SizedBox(height: 4),
                Glass(
                  radius: 18,
                  padding: const EdgeInsets.symmetric(vertical: 4),
                  child: Column(
                    children: [
                      _buildRow(
                        c,
                        label: 'Clear conversations',
                        danger: true,
                        onTap: _confirmClearConversations,
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 24),

                // ── About ──────────────────────────────────────────────
                const SectionLabel(label: 'About'),
                const SizedBox(height: 4),
                Glass(
                  radius: 18,
                  padding: const EdgeInsets.symmetric(vertical: 4),
                  child: Column(
                    children: [
                      _buildRow(
                        c,
                        label: 'View on GitHub',
                        chevron: true,
                        onTap: _openGitHub,
                      ),
                      _divider(c),
                      _buildRow(
                        c,
                        label: 'Version',
                        value: '1.4.0',
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
      ),
    );
  }

  Widget _buildRow(
    AshColors c, {
    required String label,
    String? sublabel,
    String? value,
    bool? toggle,
    ValueChanged<bool>? onToggle,
    bool chevron = false,
    bool danger = false,
    VoidCallback? onTap,
  }) {
    final textColor = danger ? Colors.redAccent : c.text;

    final content = Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 13),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  label,
                  style: TextStyle(
                    color: textColor,
                    fontSize: 15,
                  ),
                ),
                if (sublabel != null) ...[
                  const SizedBox(height: 2),
                  Text(
                    sublabel,
                    style: TextStyle(color: c.textDim, fontSize: 12),
                  ),
                ],
              ],
            ),
          ),
          if (value != null)
            Text(
              value,
              style: TextStyle(
                color: c.textMuted,
                fontSize: 13,
              ),
            ),
          if (toggle != null) ...[
            const SizedBox(width: 8),
            SizedBox(
              width: 44,
              height: 26,
              child: FittedBox(
                child: Switch(
                  value: toggle,
                  onChanged: onToggle,
                  activeThumbColor: c.accent,
                  activeTrackColor: c.accentSoft,
                ),
              ),
            ),
          ],
          if (chevron) ...[
            const SizedBox(width: 4),
            Icon(
              Icons.chevron_right,
              size: 20,
              color: c.textDim,
            ),
          ],
        ],
      ),
    );
    if (onTap == null) return content;
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(12),
      child: content,
    );
  }

  /// Bottom-sheet picker showing every known [LlmModel] with its install
  /// state. Installed → tap to switch (closes engine, host shows banner).
  /// Not installed → tap to start the download (host pushes the download
  /// stage). The current active model is marked and disabled.
  void _openModelPicker() async {
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
                    isActive: m == widget.activeLlmModel,
                    isInstalled: widget.installedLlmModels.contains(m),
                    onTap: () => Navigator.of(sheetContext).pop(m),
                    onDelete: () async {
                      Navigator.of(sheetContext).pop(); // close sheet first
                      await _confirmDelete(m);
                    },
                  ),
              ],
            ),
          ),
        ),
      ),
    );
    if (!mounted || selected == null) return;
    if (selected == widget.activeLlmModel) return; // no-op
    if (widget.installedLlmModels.contains(selected)) {
      widget.onSwitchLlmModel(selected);
    } else {
      widget.onInstallLlmModel(selected);
    }
  }

  /// Confirm + delegate model deletion. Reachable for any installed
  /// variant — the host handles the "deleting the active one" case by
  /// switching to another installed model first, or returning to ModelPick
  /// if there's nothing left to fall back to.
  Future<void> _confirmDelete(LlmModel m) async {
    final isActive = m == widget.activeLlmModel;
    final hasFallback = widget.installedLlmModels.any((other) => other != m);
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
    if (confirm == true) widget.onDeleteLlmModel(m);
  }

  Widget _buildSpeechRateRow(AshColors c) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  'Speech rate',
                  style: TextStyle(color: c.text, fontSize: 15),
                ),
              ),
              Text(
                _rateLabel(widget.voiceService.speechRate),
                style: TextStyle(color: c.textMuted, fontSize: 13),
              ),
            ],
          ),
          SliderTheme(
            data: SliderThemeData(
              activeTrackColor: c.accent,
              inactiveTrackColor: c.border,
              thumbColor: c.accent,
              overlayColor: c.accentSoft,
              trackHeight: 3,
            ),
            child: Slider(
              value: widget.voiceService.speechRate.clamp(0.3, 0.7),
              min: 0.3,
              max: 0.7,
              divisions: 8,
              onChanged: (v) async {
                await widget.voiceService.setSpeechRate(v);
                if (mounted) setState(() {});
              },
            ),
          ),
        ],
      ),
    );
  }

  String _rateLabel(double rate) {
    if (rate <= 0.35) return 'Slow';
    if (rate <= 0.45) return 'Relaxed';
    if (rate <= 0.55) return 'Natural';
    if (rate <= 0.65) return 'Brisk';
    return 'Fast';
  }

  Widget _buildListeningPatienceRow(AshColors c) {
    final seconds = widget.voiceService.listeningPatience.inSeconds
        .clamp(3, 10)
        .toDouble();
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  'Listening patience',
                  style: TextStyle(color: c.text, fontSize: 15),
                ),
              ),
              Text(
                '${seconds.toInt()}s · ${_patienceLabel(seconds.toInt())}',
                style: TextStyle(color: c.textMuted, fontSize: 13),
              ),
            ],
          ),
          SliderTheme(
            data: SliderThemeData(
              activeTrackColor: c.accent,
              inactiveTrackColor: c.border,
              thumbColor: c.accent,
              overlayColor: c.accentSoft,
              trackHeight: 3,
            ),
            child: Slider(
              value: seconds,
              min: 3,
              max: 10,
              divisions: 7,
              onChanged: (v) async {
                await widget.voiceService
                    .setListeningPatience(Duration(seconds: v.toInt()));
                if (mounted) setState(() {});
              },
            ),
          ),
          Padding(
            padding: const EdgeInsets.only(top: 2),
            child: Text(
              'How long the live-mode mic waits in silence before sending '
              'your turn to the model.',
              style: TextStyle(color: c.textDim, fontSize: 12),
            ),
          ),
        ],
      ),
    );
  }

  String _patienceLabel(int seconds) {
    if (seconds <= 3) return 'Quick';
    if (seconds <= 5) return 'Normal';
    if (seconds <= 7) return 'Patient';
    return 'Very patient';
  }

  /// Show a confirm dialog before wiping conversation history.
  void _confirmClearConversations() async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Clear conversations?'),
        content: const Text(
          'This deletes every chat from this device. Cannot be undone.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(true),
            child: const Text(
              'Clear',
              style: TextStyle(color: Colors.redAccent),
            ),
          ),
        ],
      ),
    );
    if (confirm == true) widget.onClearConversations();
  }

  /// Open the GitHub repo in the system browser.
  void _openGitHub() async {
    final uri = Uri.parse('https://github.com/RaccoonOnion/ash');
    if (!await launchUrl(uri, mode: LaunchMode.externalApplication)) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Could not open browser')),
      );
    }
  }

  /// MTP / speculative-decoding row. Boolean switch with a sublabel that
  /// names the engine-reload cost so the user isn't surprised by the ~5s
  /// banner after flipping it.
  Widget _buildSpeculativeRow(AshColors c) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 12),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  'Speculative decoding',
                  style: TextStyle(color: c.text, fontSize: 15),
                ),
                const SizedBox(height: 2),
                Text(
                  'Multi-Token Prediction. ~1.5–2× faster generation when '
                  'the model supports it. Reloads engine.',
                  style: TextStyle(color: c.textDim, fontSize: 12),
                ),
              ],
            ),
          ),
          const SizedBox(width: 8),
          SizedBox(
            width: 44,
            height: 26,
            child: FittedBox(
              child: Switch(
                value: widget.speculativeDecoding,
                onChanged: widget.onSpeculativeDecodingChanged,
                activeThumbColor: c.accent,
                activeTrackColor: c.accentSoft,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildAcceleratorRow(AshColors c) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  'Accelerator',
                  style: TextStyle(color: c.text, fontSize: 15),
                ),
              ),
              Text(
                'Reloads engine',
                style: TextStyle(color: c.textDim, fontSize: 12),
              ),
            ],
          ),
          const SizedBox(height: 8),
          _AcceleratorToggle(
            value: widget.accelerator,
            onChanged: widget.onAcceleratorChanged,
          ),
        ],
      ),
    );
  }

  Widget _divider(AshColors c) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: Divider(height: 1, color: c.border),
    );
  }
}

/// One row in the model-picker sheet. Shows the variant's name, badge for
/// active/installed/needs-download, size, and the blurb. Tap selects the row.
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
  /// Tap on the trash icon. Only rendered for installed-but-inactive
  /// variants (active models can't be deleted — user must switch first to
  /// avoid leaving the engine in a no-active-model state).
  final VoidCallback? onDelete;

  @override
  Widget build(BuildContext context) {
    final c = ash(context);
    final (badgeLabel, badgeColor) = isActive
        ? ('Active', c.accent)
        : isInstalled
            ? ('Installed', c.text)
            : ('Download', c.textDim);
    // Show delete on any installed variant — including the active one.
    // The host's delete handler does a safe auto-switch to another
    // installed model first (or routes to ModelPick if this is the last
    // one) so the engine never points at a missing file.
    final showTrash = isInstalled && onDelete != null;
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
            if (showTrash)
              IconButton(
                icon: Icon(Icons.delete_outline, size: 20, color: c.textMuted),
                onPressed: onDelete,
                tooltip: 'Delete download',
                visualDensity: VisualDensity.compact,
              )
            else if (isActive)
              Icon(Icons.check, size: 18, color: c.accent)
            else if (!isInstalled)
              Icon(Icons.download_outlined, size: 18, color: c.textMuted),
          ],
        ),
      ),
    );
  }
}

/// Two-segment pill switch for GPU vs CPU. Looks like a small dual-button
/// row so it reads as a hard choice (not an on/off boolean).
class _AcceleratorToggle extends StatelessWidget {
  const _AcceleratorToggle({required this.value, required this.onChanged});

  final AcceleratorChoice value;
  final ValueChanged<AcceleratorChoice> onChanged;

  @override
  Widget build(BuildContext context) {
    final c = ash(context);
    return Container(
      decoration: BoxDecoration(
        color: c.surfaceElev,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: c.border, width: 0.5),
      ),
      padding: const EdgeInsets.all(3),
      child: Row(
        children: [
          _segment(c, label: 'GPU', choice: AcceleratorChoice.gpu),
          _segment(c, label: 'CPU', choice: AcceleratorChoice.cpu),
        ],
      ),
    );
  }

  Widget _segment(
    AshColors c, {
    required String label,
    required AcceleratorChoice choice,
  }) {
    final selected = value == choice;
    return Expanded(
      child: GestureDetector(
        onTap: () {
          if (selected) return;
          onChanged(choice);
        },
        child: Container(
          height: 34,
          alignment: Alignment.center,
          decoration: BoxDecoration(
            color: selected ? c.accent : Colors.transparent,
            borderRadius: BorderRadius.circular(7),
          ),
          child: Text(
            label,
            style: TextStyle(
              color: selected ? Colors.white : c.text,
              fontSize: 13,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
      ),
    );
  }
}

