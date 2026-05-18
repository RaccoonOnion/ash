import 'package:flutter/material.dart';
import '../services/inference_settings.dart';
import '../services/llm_model.dart';
import '../theme/ash_theme.dart';
import 'glass.dart';

/// Bottom sheet for per-conversation inference controls (RAG toggle +
/// temperature / top-K / top-P). Pushed from the chat header's overflow menu.
class ChatSettingsSheet extends StatefulWidget {
  const ChatSettingsSheet({
    super.key,
    required this.initial,
    required this.onChanged,
    this.activeLlmModel,
    this.onOpenModelPicker,
  });

  final InferenceSettings initial;
  final ValueChanged<InferenceSettings> onChanged;
  /// Currently-active LLM variant. Surfaced as a tappable row at the top
  /// of the sheet so the user can switch model without leaving the chat
  /// surface. Null falls back to hiding the row (used by callers that
  /// haven't wired the picker yet).
  final LlmModel? activeLlmModel;
  /// Open the global model-picker overlay. The sheet closes itself before
  /// invoking the callback. Implementations typically route to
  /// SettingsScreen which already owns the picker UI.
  final VoidCallback? onOpenModelPicker;

  @override
  State<ChatSettingsSheet> createState() => _ChatSettingsSheetState();
}

class _ChatSettingsSheetState extends State<ChatSettingsSheet> {
  late InferenceSettings _current = widget.initial;

  void _update(InferenceSettings next) {
    setState(() => _current = next);
    widget.onChanged(next);
  }

  @override
  Widget build(BuildContext context) {
    final c = ash(context);
    return SafeArea(
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
                    'Inference settings (this chat)',
                    style: TextStyle(
                      color: c.text,
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 8),
              if (widget.activeLlmModel != null &&
                  widget.onOpenModelPicker != null) ...[
                _modelRow(c),
                _divider(c),
              ],
              _ragRow(c),
              _divider(c),
              _thinkingRow(c),
              _divider(c),
              _slider(
                c,
                label: 'Temperature',
                value: _current.temperature,
                min: 0.0,
                max: 1.5,
                decimals: 2,
                onChanged: (v) => _update(_current.copyWith(temperature: v)),
              ),
              _divider(c),
              _slider(
                c,
                label: 'Top-K',
                value: _current.topK.toDouble(),
                min: 1,
                max: 80,
                decimals: 0,
                onChanged: (v) =>
                    _update(_current.copyWith(topK: v.round())),
              ),
              _divider(c),
              _slider(
                c,
                label: 'Top-P',
                value: _current.topP,
                min: 0.1,
                max: 1.0,
                decimals: 2,
                onChanged: (v) => _update(_current.copyWith(topP: v)),
              ),
              _divider(c),
              _slider(
                c,
                label: 'Max tokens',
                sublabel: 'Engine context (input + history + reply). '
                    'Engine silently truncates older history when full; '
                    'changes reload the engine.',
                value: _current.maxTokens.toDouble(),
                min: 2000,
                max: 32000,
                decimals: 0,
                step: 500,
                onChanged: (v) =>
                    _update(_current.copyWith(maxTokens: v.round())),
              ),
            ],
          ),
        ),
      ),
    );
  }

  /// Tappable row showing the active LLM. Tap closes this sheet and hands
  /// off to the host's model picker (typically the Settings overlay which
  /// already owns the full installed-models list). Putting this here means
  /// the user never has to leave a chat to swap models — one tap into
  /// settings, one tap on a row, one tap back, done.
  Widget _modelRow(AshColors c) {
    final m = widget.activeLlmModel!;
    return InkWell(
      onTap: () {
        Navigator.of(context).maybePop();
        widget.onOpenModelPicker?.call();
      },
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        child: Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    'Model',
                    style: TextStyle(color: c.text, fontSize: 15),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    '${m.displayName} · ${m.sizeGB.toStringAsFixed(1)} GB',
                    style: TextStyle(color: c.textDim, fontSize: 12),
                  ),
                ],
              ),
            ),
            Icon(Icons.chevron_right, color: c.textDim, size: 20),
          ],
        ),
      ),
    );
  }

  Widget _thinkingRow(AshColors c) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  'Thinking mode',
                  style: TextStyle(color: c.text, fontSize: 15),
                ),
                const SizedBox(height: 2),
                Text(
                  'Let the model reason step-by-step before answering. '
                  'Slower but better on harder questions.',
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
                value: _current.isThinking,
                onChanged: (v) => _update(_current.copyWith(isThinking: v)),
                activeThumbColor: c.accent,
                activeTrackColor: c.accentSoft,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _ragRow(AshColors c) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  'Use reference material (RAG)',
                  style: TextStyle(color: c.text, fontSize: 15),
                ),
                const SizedBox(height: 2),
                Text(
                  'Retrieve from your knowledge packs when answering',
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
                value: _current.useRag,
                onChanged: (v) => _update(_current.copyWith(useRag: v)),
                activeThumbColor: c.accent,
                activeTrackColor: c.accentSoft,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _slider(
    AshColors c, {
    required String label,
    String? sublabel,
    required double value,
    required double min,
    required double max,
    required int decimals,
    required ValueChanged<double> onChanged,
    double? step,
  }) {
    final display = decimals == 0
        ? value.round().toString()
        : value.toStringAsFixed(decimals);
    final divs = step != null
        ? ((max - min) / step).round()
        : (decimals == 0 ? (max - min).round() : ((max - min) * 20).round());
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 4, 16, 4),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  label,
                  style: TextStyle(color: c.text, fontSize: 15),
                ),
              ),
              Text(
                display,
                style: TextStyle(
                  color: c.accent,
                  fontSize: 13,
                  fontFeatures: const [FontFeature.tabularFigures()],
                ),
              ),
            ],
          ),
          if (sublabel != null) ...[
            const SizedBox(height: 2),
            Text(
              sublabel,
              style: TextStyle(color: c.textDim, fontSize: 11),
            ),
          ],
          SliderTheme(
            data: SliderThemeData(
              activeTrackColor: c.accent,
              inactiveTrackColor: c.border,
              thumbColor: c.accent,
              overlayColor: c.accentSoft,
              trackHeight: 3,
            ),
            child: Slider(
              value: value.clamp(min, max),
              min: min,
              max: max,
              divisions: divs,
              onChanged: onChanged,
            ),
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
