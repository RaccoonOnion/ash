import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import '../theme/ash_theme.dart';
import '../models/chat.dart';
import '../models/pack.dart';
import '../services/context_estimator.dart';
import '../services/inference_settings.dart';
import '../services/llm_model.dart';
import '../services/voice_service.dart';
import '../widgets/chat_settings_sheet.dart';
import '../widgets/glass.dart';
import '../widgets/ash_mark.dart';
import '../widgets/buttons.dart';
import '../widgets/chat_bubble.dart';
import '../widgets/composer_attachment.dart';
import '../widgets/lens_sheet.dart';

/// Active conversation screen with real on-device inference.
class ChatScreen extends StatefulWidget {
  const ChatScreen({
    required this.chat,
    required this.installedPacks,
    required this.modelReady,
    required this.onBack,
    required this.onOpenVoice,
    required this.onOpenCamera,
    required this.onSendInChat,
    required this.onSettingsChanged,
    required this.onSettingsCommitted,
    required this.onRename,
    required this.onChangeLens,
    required this.onOpenInLibrary,
    required this.onContextTrim,
    required this.onContextExtend,
    this.activeLlmModel,
    this.onOpenModelPicker,
    this.voiceService,
    this.pendingImageBytes,
    this.onClearPendingImage,
    super.key,
  });

  final Chat chat;
  /// All packs currently installed. Used to populate the lens-pill sheet.
  final List<Pack> installedPacks;
  /// Whether the on-disk model file is ready for inference. When false
  /// (first launch, model is still downloading), the composer drops into
  /// a muted disabled state so the user can compose ahead but can't fire
  /// a send into a half-loaded engine. The host shows the actual download
  /// banner at screen-bottom.
  final bool modelReady;
  final VoidCallback onBack;
  final VoidCallback onOpenVoice;
  final VoidCallback onOpenCamera;
  final ValueChanged<String> onSendInChat;
  /// Fires on every slider tick while the settings sheet is open. Use for
  /// live UI updates — do NOT trigger expensive engine work here.
  final ValueChanged<InferenceSettings> onSettingsChanged;
  /// Fires once when the settings sheet closes, IF anything actually changed.
  /// Use this to eagerly apply settings (engine reload + session rebuild) so
  /// the user doesn't have to wait on the first message after closing.
  final VoidCallback onSettingsCommitted;
  /// Invoked with the new title when the user renames this chat from the
  /// in-chat header. Null-safe text trimming/validation happens at the host.
  final ValueChanged<String> onRename;
  /// Invoked with the new lens selection from the lens-pill bottom sheet.
  /// `null` means "all installed packs".
  final ValueChanged<Set<String>?> onChangeLens;
  /// Invoked when the user taps "Open in Library" on a citation chip's
  /// detail sheet. Receives the packId of the cited source and the
  /// chunkId so the Library reader can scroll directly to that
  /// proposition. `chunkId` may be empty for older persisted messages —
  /// the reader treats that as "open at the top of the pack".
  final void Function(String packId, String chunkId) onOpenInLibrary;
  /// Tap-handler for the context-load banner's "Trim older" button.
  /// The host drops the oldest ~30% of [chat.messages] and resets the
  /// inference service's chat session so the next query rebuilds
  /// against the shorter history. Returns the count of dropped
  /// messages (0 if there weren't enough turns to safely trim) so the
  /// chat screen can show a confirmation toast.
  final Future<int> Function() onContextTrim;
  /// Tap-handler for the context-load banner's "Extend memory" button.
  /// The host bumps [chat.inferenceSettings.maxTokens] by 2048 (capped
  /// at 32k) and triggers the eager engine reload. Returns the new
  /// maxTokens value so the chat screen can show a confirmation toast.
  final Future<int> Function() onContextExtend;
  /// Active LLM variant — passed into the chat-settings sheet so the user
  /// can swap model without leaving the chat surface.
  final LlmModel? activeLlmModel;
  /// Open the global model-picker overlay (e.g. the Settings screen).
  /// Invoked from the chat-settings sheet's Model row.
  final VoidCallback? onOpenModelPicker;
  final VoiceService? voiceService;
  final Uint8List? pendingImageBytes;
  final VoidCallback? onClearPendingImage;

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final _draftController = TextEditingController();
  final _scrollController = ScrollController();

  // The widget.chat.messages list is mutated in place by AshApp during
  // streaming, so didUpdateWidget can't compare oldWidget vs widget — both
  // hold the same list reference. Snapshot count + last text length here
  // and compare against the live widget on every rebuild.
  int _lastMsgCount = 0;
  int _lastTextLen = 0;
  // Keyboard-inset snapshot. When the iOS keyboard appears, viewInsets.bottom
  // changes which causes a rebuild but message content is unchanged — without
  // this we'd skip scrolling and the last message would sit hidden behind the
  // newly-shown keyboard.
  double _lastBottomInset = 0;
  // Inline dictation: mic icon fills the text field with live partials. The
  // user can edit + send manually. Distinct from full-screen Live mode.
  bool _isDictating = false;
  String _dictationBase = '';
  StreamSubscription<String>? _dictationPartialSub;
  StreamSubscription<String>? _dictationFinalSub;
  // Local toast — we render it ourselves instead of ScaffoldMessenger because
  // the widget tree has no Scaffold and snackbars don't show up reliably here.
  String? _toast;
  Timer? _toastTimer;

  @override
  void initState() {
    super.initState();
    // One-shot composer prefill from example-prompt chips on the home
    // screen. Consume and clear so a re-mount (e.g. tab switch) doesn't
    // re-prefill on top of whatever the user has typed since.
    final seed = widget.chat.initialComposerText;
    if (seed != null && seed.isNotEmpty) {
      _draftController.text = seed;
      _draftController.selection =
          TextSelection.collapsed(offset: seed.length);
      widget.chat.initialComposerText = null;
    }
    // Listen so the context-load banner re-evaluates as the user types.
    // The estimator is cheap (one walk over message texts + draft); the
    // setState rebuild only re-runs build(), and the ListView is lazy.
    _draftController.addListener(_onDraftChanged);
  }

  void _onDraftChanged() {
    if (mounted) setState(() {});
  }

  @override
  void dispose() {
    _draftController.removeListener(_onDraftChanged);
    _draftController.dispose();
    _scrollController.dispose();
    _dictationPartialSub?.cancel();
    _dictationFinalSub?.cancel();
    _toastTimer?.cancel();
    super.dispose();
  }

  Future<void> _toggleDictation() async {
    final service = widget.voiceService;
    if (service == null) return;
    if (_isDictating) {
      await service.stopAndTranscribe();
      await _dictationPartialSub?.cancel();
      await _dictationFinalSub?.cancel();
      _dictationPartialSub = null;
      _dictationFinalSub = null;
      if (mounted) setState(() => _isDictating = false);
      return;
    }
    final status = await service.getStatus();
    if (!mounted) return;
    if (!status.sttReady || !status.micPermitted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text(
            'Enable Microphone AND Speech Recognition in iOS Settings.',
          ),
        ),
      );
      return;
    }
    FocusScope.of(context).unfocus();
    _dictationBase = _draftController.text;
    if (_dictationBase.isNotEmpty && !_dictationBase.endsWith(' ')) {
      _dictationBase = '$_dictationBase ';
    }
    _dictationPartialSub = service.partialStream.listen((partial) {
      _draftController.text = '$_dictationBase$partial';
      _draftController.selection = TextSelection.collapsed(
        offset: _draftController.text.length,
      );
    });
    _dictationFinalSub = service.finalStream.listen((finalText) async {
      _draftController.text = '$_dictationBase$finalText';
      _draftController.selection = TextSelection.collapsed(
        offset: _draftController.text.length,
      );
      // Final result means iOS auto-stopped (silence). End the dictation
      // toggle visually; user can tap mic again to extend.
      if (mounted) setState(() => _isDictating = false);
      await _dictationPartialSub?.cancel();
      await _dictationFinalSub?.cancel();
      _dictationPartialSub = null;
      _dictationFinalSub = null;
    });
    try {
      await service.startRecording();
      if (mounted) setState(() => _isDictating = true);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Voice: $e')),
        );
      }
    }
  }

  /// Called from build() on every rebuild. Cheap when nothing's changed.
  /// Schedules a post-frame scroll to the bottom whenever the message count,
  /// the last-message text length, OR the keyboard inset changes. Watching
  /// the inset is what keeps the latest message visible when the keyboard
  /// appears mid-conversation — the rebuild from MediaQuery alone wouldn't
  /// otherwise trigger a scroll.
  void _maybeScrollToBottomOnContentGrowth(BuildContext context) {
    final msgs = widget.chat.messages;
    final count = msgs.length;
    final lastTextLen = msgs.isEmpty ? 0 : msgs.last.text.length;
    final bottomInset = MediaQuery.of(context).viewInsets.bottom;
    final grew = count != _lastMsgCount;
    final lastTextGrew = !grew && lastTextLen != _lastTextLen;
    final insetChanged = bottomInset != _lastBottomInset;
    if (grew || lastTextGrew || insetChanged) {
      _lastMsgCount = count;
      _lastTextLen = lastTextLen;
      _lastBottomInset = bottomInset;
      _scrollToBottom(animate: grew || insetChanged);
    }
  }

  bool get _isGenerating {
    final msgs = widget.chat.messages;
    if (msgs.isEmpty) return false;
    return msgs.last.streaming;
  }

  void _handleSend() {
    final text = _draftController.text.trim();
    if (_isGenerating) return;
    if (!widget.modelReady) return; // composer button is muted; defense in depth
    if (text.isEmpty && widget.pendingImageBytes == null) return;
    _draftController.clear();
    widget.onSendInChat(text);
    _scrollToBottom();
  }

  void _openSettingsSheet() async {
    // Snapshot before opening so we can diff after the user closes the sheet
    // and surface a single summary toast describing what will apply on the
    // next message. The sheet itself still mutates settings live so streaming
    // changes work, but per-tick toasts would be noise.
    final before = widget.chat.inferenceSettings;
    await showModalBottomSheet<void>(
      context: context,
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      builder: (sheetContext) => ChatSettingsSheet(
        initial: before,
        activeLlmModel: widget.activeLlmModel,
        onOpenModelPicker: widget.onOpenModelPicker,
        onChanged: (next) {
          // Mutate in place + notify the host so it can rebuild and pass the
          // fresh settings to the next inference call. The model's chat
          // session is rebuilt automatically when the sampling signature
          // changes (see gemma_inference_service.dart).
          widget.chat.inferenceSettings = next;
          widget.onSettingsChanged(next);
        },
      ),
    );
    if (!mounted) return;
    final after = widget.chat.inferenceSettings;
    final msg = _settingsChangeSummary(before, after);
    if (msg != null) {
      _showSettingsToast(msg);
      // Tell the host to kick off the eager apply: it'll show the banner
      // and call the service to do the engine/session rebuild NOW so the
      // user's next message streams without a leading pause.
      widget.onSettingsCommitted();
    }
  }

  /// Returns a one-line description of what the user changed and what cost
  /// that implies on the next message. Null if nothing material changed.
  /// Phrased so the user knows changes are *deferred* — they take effect on
  /// the next send, not retroactively.
  String? _settingsChangeSummary(InferenceSettings before, InferenceSettings after) {
    final changes = <String>[];
    if (before.useRag != after.useRag) {
      changes.add(after.useRag ? 'RAG on' : 'RAG off');
    }
    if (before.temperature != after.temperature) {
      changes.add('temp ${after.temperature.toStringAsFixed(2)}');
    }
    if (before.topK != after.topK) changes.add('topK ${after.topK}');
    if (before.topP != after.topP) {
      changes.add('topP ${after.topP.toStringAsFixed(2)}');
    }
    final maxTokensChanged = before.maxTokens != after.maxTokens;
    if (maxTokensChanged) changes.add('maxTokens ${after.maxTokens}');
    if (before.isThinking != after.isThinking) {
      changes.add(after.isThinking ? 'thinking on' : 'thinking off');
    }
    if (changes.isEmpty) return null;
    // The host kicks off the eager apply right after this toast fires, so
    // phrase the cost in present tense ("Reloading…") not future
    // ("Will reload on next message…"). The banner shows the same status
    // for the duration of the reload; the toast is just commit confirmation.
    final cost = maxTokensChanged
        ? 'Reloading engine (~5s, ~30s with vision).'
        : 'Refreshing chat session.';
    return '${changes.join(', ')}. $cost';
  }

  void _showSettingsToast(String message) {
    _toastTimer?.cancel();
    setState(() => _toast = message);
    _toastTimer = Timer(const Duration(seconds: 5), () {
      if (mounted) setState(() => _toast = null);
    });
  }

  Future<void> _runTrim() async {
    final dropped = await widget.onContextTrim();
    if (!mounted) return;
    if (dropped == 0) {
      _showSettingsToast('Not enough history to trim yet.');
    } else {
      _showSettingsToast(
          'Trimmed $dropped older message${dropped == 1 ? '' : 's'}.');
    }
  }

  Future<void> _runExtend() async {
    final before = widget.chat.inferenceSettings.maxTokens;
    final next = await widget.onContextExtend();
    if (!mounted) return;
    if (next == before) {
      _showSettingsToast('Memory is already at the maximum (32k).');
    } else {
      _showSettingsToast(
          'Memory extended to $next tokens. Reloading engine.');
    }
  }

  /// Banner above the composer warning the user that the chat is
  /// approaching the model's context window. Below 80% load it returns
  /// an empty widget. At 80-95% it shows a muted notice with two
  /// actions; at 95%+ the same row tinted orange to read as a stronger
  /// warning. Estimate is char-based — see [estimateContextLoad] in
  /// [context_estimator.dart] for the math.
  Widget _buildContextLoadBanner(AshColors c) {
    final est = estimateContextLoad(widget.chat, _draftController.text);
    if (est.fraction < 0.80) return const SizedBox.shrink();
    final isOverflow = est.fraction >= 1.0;
    final isCritical = est.fraction >= 0.95;
    final tint = isCritical ? Colors.orangeAccent : c.accent;
    final atMax = widget.chat.inferenceSettings.maxTokens >= 32000;
    final headline = isOverflow
        ? 'Memory exceeded (~${est.pct}%) — earliest turns will be '
            'dropped by the engine.'
        : (isCritical
            ? 'Conversation memory is almost full (~${est.pct}%).'
            : 'Memory ~${est.pct}% full — old messages may be forgotten.');
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Container(
        padding: const EdgeInsets.fromLTRB(12, 10, 12, 10),
        decoration: BoxDecoration(
          color: c.surfaceElev,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(
            color: tint.withValues(alpha: 0.45),
            width: 0.8,
          ),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Row(
              children: [
                Icon(
                  isCritical
                      ? Icons.warning_amber_rounded
                      : Icons.info_outline,
                  color: tint,
                  size: 16,
                ),
                const SizedBox(width: 6),
                Expanded(
                  child: Text(
                    headline,
                    style: TextStyle(
                      color: c.text,
                      fontSize: 12.5,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                _ContextActionBtn(
                  c: c,
                  tint: tint,
                  label: 'Trim older',
                  onTap: _runTrim,
                ),
                const SizedBox(width: 8),
                _ContextActionBtn(
                  c: c,
                  tint: tint,
                  label: atMax ? 'At max' : 'Extend memory',
                  onTap: atMax ? null : _runExtend,
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  /// Opens the lens-scope bottom sheet. The user can pick a subset of
  /// installed packs to retrieve from, or clear the selection to retrieve
  /// from all of them. The result is delivered to the host via
  /// `widget.onChangeLens` which mutates the chat + service in lockstep.
  Future<void> _openLensSheet() async {
    final result = await showModalBottomSheet<LensResult>(
      context: context,
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      builder: (sheetContext) => LensSheet(
        installedPacks: widget.installedPacks,
        initialLens: widget.chat.lensPackIds,
      ),
    );
    if (result == null) return;
    widget.onChangeLens(result.lensPackIds);
  }

  /// Inline rename — tapping the title in the header pops a dialog so the
  /// user doesn't have to back out to the chat list to change the name.
  Future<void> _promptRename() async {
    final c = ash(context);
    final controller = TextEditingController(text: widget.chat.title);
    final result = await showDialog<String>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: c.surfaceElev,
        title: Text('Rename conversation',
            style: TextStyle(color: c.text, fontSize: 16)),
        content: TextField(
          controller: controller,
          autofocus: true,
          maxLength: 60,
          style: TextStyle(color: c.text, fontSize: 15),
          decoration: InputDecoration(
            hintText: 'Title',
            hintStyle: TextStyle(color: c.textDim),
            border: const OutlineInputBorder(),
            counterStyle: TextStyle(color: c.textDim),
          ),
          onSubmitted: (v) => Navigator.of(ctx).pop(v),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(),
            child: Text('Cancel', style: TextStyle(color: c.textMuted)),
          ),
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(controller.text),
            child: Text('Save', style: TextStyle(color: c.accent)),
          ),
        ],
      ),
    );
    controller.dispose();
    if (result == null) return;
    final trimmed = result.trim();
    if (trimmed.isEmpty || trimmed == widget.chat.title) return;
    widget.onRename(trimmed);
  }

  void _scrollToBottom({bool animate = true}) {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!_scrollController.hasClients) return;
      final target = _scrollController.position.maxScrollExtent;
      if (animate) {
        _scrollController.animateTo(
          target,
          duration: const Duration(milliseconds: 200),
          curve: Curves.easeOut,
        );
      } else {
        // Jump (no animation) during token streaming — animating each frame
        // fights the next jump and looks janky.
        _scrollController.jumpTo(target);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final c = ash(context);
    final messages = widget.chat.messages;
    final hasMessages = messages.isNotEmpty;

    // Re-evaluate scroll position every build. AshApp streams tokens by
    // calling setState which rebuilds this widget; we follow that growth so
    // the last assistant line stays in view. Also follow keyboard
    // appearance so the last message doesn't get pinned behind it.
    _maybeScrollToBottomOnContentGrowth(context);

    // Tap-outside-text dismisses the iOS keyboard (carried over from main's
    // 13b01c2). Wrap our existing Stack so the gesture detector covers all
    // non-input areas of the screen.
    return GestureDetector(
      onTap: () => FocusScope.of(context).unfocus(),
      behavior: HitTestBehavior.translucent,
      child: Stack(
      children: [
        // Main content
        SafeArea(
          child: Column(
            children: [
              // Header
              Padding(
                padding: const EdgeInsets.fromLTRB(8, 8, 8, 0),
                child: Row(
                  children: [
                    IconBtn(
                      icon: Icons.arrow_back_ios_new,
                      onTap: widget.onBack,
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          // Title is its own gesture target so the lens pill
                          // below stays independently tappable.
                          GestureDetector(
                            onTap: _promptRename,
                            behavior: HitTestBehavior.opaque,
                            child: Row(
                              children: [
                                Flexible(
                                  child: Text(
                                    widget.chat.title,
                                    maxLines: 1,
                                    overflow: TextOverflow.ellipsis,
                                    style: TextStyle(
                                      color: c.text,
                                      fontSize: 15.5,
                                      fontWeight: FontWeight.w700,
                                    ),
                                  ),
                                ),
                                const SizedBox(width: 6),
                                // Faint edit hint so the tap affordance is
                                // discoverable without making the header
                                // visually noisy.
                                Icon(
                                  Icons.edit_outlined,
                                  color: c.textDim,
                                  size: 13,
                                ),
                              ],
                            ),
                          ),
                          const SizedBox(height: 2),
                          Row(
                            children: [
                              _LensPill(
                                lensPackIds: widget.chat.lensPackIds,
                                installedCount: widget.installedPacks.length,
                                onTap: _openLensSheet,
                              ),
                              const SizedBox(width: 8),
                              Container(
                                width: 6,
                                height: 6,
                                decoration: BoxDecoration(
                                  color: c.success,
                                  shape: BoxShape.circle,
                                ),
                              ),
                              const SizedBox(width: 4),
                              Text(
                                'On-device',
                                style: TextStyle(
                                  color: c.textMuted,
                                  fontSize: 12,
                                ),
                              ),
                              // Subtle "Thinking" chip — visible only when
                              // the per-chat setting is on, so the user can
                              // tell at a glance whether replies will carry
                              // a <think> reasoning trace. Intentionally
                              // low-contrast: a state indicator, not a
                              // call to action.
                              if (widget.chat.inferenceSettings.isThinking) ...[
                                const SizedBox(width: 8),
                                Container(
                                  padding: const EdgeInsets.symmetric(
                                      horizontal: 6, vertical: 2),
                                  decoration: BoxDecoration(
                                    color: c.accent.withValues(alpha: 0.14),
                                    borderRadius: BorderRadius.circular(6),
                                    border: Border.all(
                                      color: c.accent.withValues(alpha: 0.3),
                                      width: 0.5,
                                    ),
                                  ),
                                  child: Row(
                                    mainAxisSize: MainAxisSize.min,
                                    children: [
                                      Icon(Icons.psychology_outlined,
                                          size: 11, color: c.accent),
                                      const SizedBox(width: 3),
                                      Text(
                                        'Thinking',
                                        style: TextStyle(
                                          color: c.accent,
                                          fontSize: 11,
                                          fontWeight: FontWeight.w600,
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ],
                            ],
                          ),
                        ],
                      ),
                    ),
                    IconBtn(
                      icon: Icons.tune,
                      onTap: _openSettingsSheet,
                    ),
                  ],
                ),
              ),

              // Messages or empty state
              Expanded(
                child: hasMessages
                    ? ListView.separated(
                        controller: _scrollController,
                        // Bottom padding has to cover BOTH the composer (~76)
                        // AND the iOS keyboard when it's up. Without
                        // viewInsets.bottom here, maxScrollExtent stops short
                        // of the keyboard and the last message gets pinned
                        // behind the composer — looking exactly like
                        // "auto-scroll doesn't work" even though scrollTo
                        // is firing correctly.
                        padding: EdgeInsets.fromLTRB(
                          16,
                          16,
                          16,
                          100 + MediaQuery.of(context).viewInsets.bottom,
                        ),
                        itemCount: messages.length,
                        separatorBuilder: (_, __) =>
                            const SizedBox(height: 14),
                        itemBuilder: (context, index) {
                          final msg = messages[index];
                          if (msg.role == 'user') {
                            return UserBubble(text: msg.text, imageBytes: msg.imageBytes);
                          } else {
                            return AssistantBubble(
                              text: msg.text,
                              streaming: msg.streaming,
                              sources: msg.sources,
                              onOpenInLibrary: widget.onOpenInLibrary,
                            );
                          }
                        },
                      )
                    : Center(
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            AshMark(size: 48, glow: true),
                            const SizedBox(height: 16),
                            Text(
                              "What's happening?",
                              style: TextStyle(
                                color: c.textMuted,
                                fontSize: 18,
                                fontWeight: FontWeight.w500,
                              ),
                            ),
                          ],
                        ),
                      ),
              ),
            ],
          ),
        ),

        // Bottom composer — push above the iOS keyboard when it's visible.
        // viewInsets.bottom is 0 when no keyboard, otherwise the keyboard
        // height. The +28 keeps a comfortable gap from the home indicator
        // / keyboard top edge.
        Positioned(
          left: 16,
          right: 16,
          bottom: 28 + MediaQuery.of(context).viewInsets.bottom,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              if (widget.pendingImageBytes != null) ...[
                ComposerAttachment(
                  bytes: widget.pendingImageBytes!,
                  onRemove: widget.onClearPendingImage ?? () {},
                ),
                const SizedBox(height: 8),
              ],
              _buildContextLoadBanner(c),
              Opacity(
                opacity: widget.modelReady ? 1.0 : 0.72,
                child: Glass(
                radius: 28,
                strong: true,
                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 6),
                child: Row(
                  children: [
                    // Camera button — disabled while the model is still
                    // downloading (image attach would dead-end at send).
                    IconBtn(
                      icon: Icons.camera_alt_outlined,
                      onTap: widget.modelReady ? widget.onOpenCamera : () {},
                      size: 40,
                    ),
                    const SizedBox(width: 4),
                    // Text field — stays enabled so the user can compose
                    // ahead. Hint reflects the gate.
                    Expanded(
                      child: SizedBox(
                        height: 40,
                        child: TextField(
                          controller: _draftController,
                          onSubmitted: (_) => _handleSend(),
                          style: TextStyle(
                            color: c.text,
                            fontSize: 15,
                          ),
                          decoration: InputDecoration(
                            hintText: !widget.modelReady
                                ? 'Setting up\u2026'
                                : widget.pendingImageBytes != null
                                    ? 'Add a note about this photo\u2026'
                                    : 'Reply\u2026',
                            hintStyle: TextStyle(
                              // Brighter when waiting so the message remains
                              // legible after the parent Opacity wrap dims
                              // the whole composer.
                              color: !widget.modelReady
                                  ? c.textMuted
                                  : c.textDim,
                              fontSize: 15,
                            ),
                            border: InputBorder.none,
                            contentPadding: const EdgeInsets.symmetric(
                              horizontal: 8,
                              vertical: 10,
                            ),
                            isDense: true,
                          ),
                        ),
                      ),
                    ),
                    // ChatGPT-style two-icon split:
                    //   • Mic = inline dictation (fills the text field, user
                    //     edits/sends manually).
                    //   • Waveform = full-screen Live voice mode.
                    // When there's text or a pending image, the waveform
                    // collapses into a Send button.
                    ValueListenableBuilder<TextEditingValue>(
                      valueListenable: _draftController,
                      builder: (context, value, _) {
                        final hasText = value.text.trim().isNotEmpty;
                        final hasContent =
                            hasText || widget.pendingImageBytes != null;
                        final canSend = widget.modelReady && hasContent;
                        return Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            IconBtn(
                              icon: _isDictating ? Icons.stop : Icons.mic,
                              onTap: widget.modelReady
                                  ? _toggleDictation
                                  : () {},
                              size: 40,
                              accentBg: _isDictating,
                            ),
                            const SizedBox(width: 4),
                            // When there's content to send, always show the
                            // send icon — muted (no accent bg, inert tap)
                            // when the model isn't ready yet. Showing the
                            // waveform here would confuse the user; the
                            // send "slot" should stay visually consistent
                            // with what the action will be once setup is
                            // done.
                            hasContent
                                ? IconBtn(
                                    icon: Icons.send,
                                    onTap: canSend ? _handleSend : () {},
                                    size: 40,
                                    accentBg: canSend,
                                  )
                                : IconBtn(
                                    icon: Icons.graphic_eq,
                                    onTap: widget.modelReady
                                        ? () {
                                            // Drop keyboard before opening
                                            // live mode so it doesn't peek
                                            // behind the new full-screen
                                            // surface.
                                            FocusScope.of(context).unfocus();
                                            widget.onOpenVoice();
                                          }
                                        : () {},
                                    size: 40,
                                  ),
                          ],
                        );
                      },
                    ),
                  ],
                ),
              ),
              ),
            ],
          ),
        ),
        // Local toast, lives above the composer. We render it ourselves
        // because ScaffoldMessenger snackbars don't show reliably in this
        // widget tree (no Scaffold ancestor — the app uses Material+Stack).
        if (_toast != null)
          Positioned(
            left: 16,
            right: 16,
            bottom: 100 + MediaQuery.of(context).viewInsets.bottom,
            child: IgnorePointer(
              child: Glass(
                radius: 14,
                strong: true,
                padding: const EdgeInsets.symmetric(
                    horizontal: 14, vertical: 10),
                child: Row(
                  children: [
                    Icon(Icons.tune, size: 16, color: c.accent),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        _toast!,
                        style: TextStyle(color: c.text, fontSize: 12),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
      ],
      ),
    );
  }
}

/// Header pill summarizing the chat's retrieval lens. Renders as
/// `📚 N packs ›` (or `📚 All packs ›` when null). Tapping opens the
/// [LensSheet] so the user can scope down or back up to all packs.
class _LensPill extends StatelessWidget {
  const _LensPill({
    required this.lensPackIds,
    required this.installedCount,
    required this.onTap,
  });

  final Set<String>? lensPackIds;
  final int installedCount;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final c = ash(context);
    final isAll = lensPackIds == null || lensPackIds!.isEmpty;
    final count = isAll ? installedCount : lensPackIds!.length;
    final label = isAll ? 'All $count packs' : '$count of $installedCount packs';
    return GestureDetector(
      onTap: onTap,
      behavior: HitTestBehavior.opaque,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
        decoration: BoxDecoration(
          color: c.surfaceElev,
          borderRadius: BorderRadius.circular(10),
          border: Border.all(color: c.border, width: 0.5),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.menu_book_outlined, size: 12, color: c.accent),
            const SizedBox(width: 5),
            Text(
              label,
              style: TextStyle(
                color: c.textMuted,
                fontSize: 12,
                fontWeight: FontWeight.w500,
              ),
            ),
            const SizedBox(width: 4),
            Icon(Icons.chevron_right, size: 14, color: c.textDim),
          ],
        ),
      ),
    );
  }
}

class _ContextActionBtn extends StatelessWidget {
  const _ContextActionBtn({
    required this.c,
    required this.tint,
    required this.label,
    required this.onTap,
  });

  final AshColors c;
  final Color tint;
  final String label;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final disabled = onTap == null;
    return Material(
      color: Colors.transparent,
      child: InkWell(
        borderRadius: BorderRadius.circular(8),
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
          decoration: BoxDecoration(
            color: disabled
                ? c.surfaceElev.withValues(alpha: 0.4)
                : tint.withValues(alpha: 0.16),
            borderRadius: BorderRadius.circular(8),
            border: Border.all(
              color: disabled
                  ? c.border
                  : tint.withValues(alpha: 0.5),
              width: 0.6,
            ),
          ),
          child: Text(
            label,
            style: TextStyle(
              color: disabled ? c.textDim : tint,
              fontSize: 12,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
      ),
    );
  }
}
