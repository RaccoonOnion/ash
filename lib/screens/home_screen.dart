import 'package:flutter/material.dart';
import '../theme/ash_theme.dart';
import '../models/chat.dart';
import '../utils/fuzzy.dart';
import '../widgets/glass.dart';
import '../widgets/ash_mark.dart';
import '../widgets/on_device_badge.dart';
import '../widgets/text_field.dart';

/// Home tab — chat list + "start new conversation" CTA, plus fuzzy search
/// and per-row star/rename/delete actions.
class HomeScreen extends StatefulWidget {
  const HomeScreen({
    required this.chats,
    required this.onOpenChat,
    required this.onNew,
    required this.onNewWithPrompt,
    required this.onDeleteChat,
    required this.onRenameChat,
    required this.onToggleStarChat,
    super.key,
  });

  final List<Chat> chats;
  final ValueChanged<String> onOpenChat;
  final VoidCallback onNew;
  /// Open a new chat with the composer pre-filled. Used by the empty-state
  /// example-prompt chips on a fresh install — tapping a chip drops the
  /// user into a new conversation with that question already typed, ready
  /// to send. Kept separate from [onNew] so the empty-flow doesn't have
  /// to fabricate a fake button press.
  final ValueChanged<String> onNewWithPrompt;
  final ValueChanged<String> onDeleteChat;
  final void Function(String id, String newTitle) onRenameChat;
  final ValueChanged<String> onToggleStarChat;

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _searchController = TextEditingController();
  String _query = '';

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  /// Filtered + sorted view of [widget.chats]. Starred chats float to the top;
  /// when a query is active, results are ranked by fuzzy score across title
  /// and preview.
  List<Chat> get _visibleChats {
    final q = _query.trim();
    if (q.isEmpty) {
      final starred = widget.chats.where((c) => c.starred).toList();
      final rest = widget.chats.where((c) => !c.starred).toList();
      return [...starred, ...rest];
    }
    final scored = <({Chat chat, int score})>[];
    for (final c in widget.chats) {
      final s = fuzzyScoreFields(q, [
        (text: c.title, weight: 1.0),
        (text: c.preview, weight: 0.6),
      ]);
      if (s == null) continue;
      // Tiny tiebreak so starred wins on equal score.
      scored.add((chat: c, score: s + (c.starred ? 1 : 0)));
    }
    scored.sort((a, b) => b.score.compareTo(a.score));
    return scored.map((e) => e.chat).toList();
  }

  Future<void> _confirmDelete(Chat chat) async {
    final c = ash(context);
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: c.surfaceElev,
        title: Text('Delete conversation?',
            style: TextStyle(color: c.text, fontSize: 16)),
        content: Text(
          'This will permanently remove "${chat.title}".',
          style: TextStyle(color: c.textMuted, fontSize: 14),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: Text('Cancel', style: TextStyle(color: c.textMuted)),
          ),
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(true),
            child: Text('Delete', style: TextStyle(color: Colors.redAccent)),
          ),
        ],
      ),
    );
    if (ok == true) widget.onDeleteChat(chat.id);
  }

  Future<void> _promptRename(Chat chat) async {
    final c = ash(context);
    final controller = TextEditingController(text: chat.title);
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
    if (result != null && result.trim().isNotEmpty) {
      widget.onRenameChat(chat.id, result);
    }
  }

  Future<void> _showChatMenu(Chat chat) async {
    final c = ash(context);
    final action = await showModalBottomSheet<String>(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (ctx) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
          child: Glass(
            radius: 24,
            strong: true,
            padding: const EdgeInsets.symmetric(vertical: 8),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                _MenuRow(
                  icon: chat.starred ? Icons.star : Icons.star_border,
                  label: chat.starred ? 'Unstar' : 'Star',
                  iconColor: chat.starred ? c.accent : c.text,
                  onTap: () => Navigator.of(ctx).pop('star'),
                ),
                _MenuRow(
                  icon: Icons.drive_file_rename_outline,
                  label: 'Rename',
                  iconColor: c.text,
                  onTap: () => Navigator.of(ctx).pop('rename'),
                ),
                _MenuRow(
                  icon: Icons.delete_outline,
                  label: 'Delete',
                  iconColor: Colors.redAccent,
                  labelColor: Colors.redAccent,
                  onTap: () => Navigator.of(ctx).pop('delete'),
                ),
              ],
            ),
          ),
        ),
      ),
    );
    if (!mounted) return;
    switch (action) {
      case 'star':
        widget.onToggleStarChat(chat.id);
        break;
      case 'rename':
        await _promptRename(chat);
        break;
      case 'delete':
        await _confirmDelete(chat);
        break;
    }
  }

  @override
  Widget build(BuildContext context) {
    final c = ash(context);
    final visible = _visibleChats;
    final hasQuery = _query.trim().isNotEmpty;

    return SafeArea(
      child: Column(
        children: [
          // Header
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 16, 20, 0),
            child: Row(
              children: [
                const AshMark(size: 24),
                const SizedBox(width: 8),
                Text(
                  'Ash',
                  style: TextStyle(
                    color: c.text,
                    fontSize: 22,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(width: 10),
                const OnDeviceBadge(),
              ],
            ),
          ),

          // Primary CTA — promoted to the top of the chat tab now that
          // the redundant Library card is gone (Knowledge tab in the
          // bottom nav serves that role). Strong glass + accent ring
          // around the icon make it the unmistakable next action.
          Padding(
            padding: const EdgeInsets.fromLTRB(20, 18, 20, 0),
            child: GestureDetector(
              onTap: widget.onNew,
              behavior: HitTestBehavior.opaque,
              child: Glass(
                radius: 20,
                strong: true,
                padding:
                    const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
                child: Row(
                  children: [
                    Container(
                      width: 44,
                      height: 44,
                      decoration: BoxDecoration(
                        color: c.accent,
                        shape: BoxShape.circle,
                      ),
                      alignment: Alignment.center,
                      child: const Icon(
                        Icons.add_comment_outlined,
                        color: Colors.white,
                        size: 22,
                      ),
                    ),
                    const SizedBox(width: 14),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Text(
                            'New conversation',
                            style: TextStyle(
                              color: c.text,
                              fontSize: 16,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                          const SizedBox(height: 2),
                          Text(
                            'Type, snap a photo, or speak — all on-device',
                            style: TextStyle(color: c.textDim, fontSize: 12),
                          ),
                        ],
                      ),
                    ),
                    Icon(Icons.arrow_forward_ios, color: c.textDim, size: 14),
                  ],
                ),
              ),
            ),
          ),

          // Search bar — only meaningful once the user has chats to
          // search through. On a fresh install (no chats, no query) the
          // empty-state prompts below own the screen instead.
          if (widget.chats.isNotEmpty)
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 14, 20, 0),
              child: AshTextField(
                hint: 'Search conversations',
                controller: _searchController,
                leadingIcon: Icons.search,
                onChanged: (v) => setState(() => _query = v),
              ),
            ),

          // Section label (also suppressed in the totally-empty state so
          // we don't show "RECENT" floating above example chips).
          if (widget.chats.isNotEmpty)
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 14, 20, 0),
              child: Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  hasQuery ? 'RESULTS' : 'RECENT',
                  style: TextStyle(
                    color: c.textDim,
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                    letterSpacing: 0.8,
                  ),
                ),
              ),
            ),

          // Chat list / empty state
          Expanded(
            child: visible.isEmpty
                ? (hasQuery
                    ? Center(
                        child: Padding(
                          padding:
                              const EdgeInsets.symmetric(horizontal: 32),
                          child: Text(
                            'No conversations match "$_query".',
                            textAlign: TextAlign.center,
                            style:
                                TextStyle(color: c.textDim, fontSize: 13),
                          ),
                        ),
                      )
                    : _EmptyStatePrompts(
                        c: c,
                        onPick: widget.onNewWithPrompt,
                      ))
                : ListView.separated(
                    padding: const EdgeInsets.fromLTRB(20, 8, 20, 120),
                    itemCount: visible.length,
                    separatorBuilder: (_, __) => const SizedBox(height: 8),
                    itemBuilder: (context, index) {
                      final chat = visible[index];
                      return _ChatRow(
                        chat: chat,
                        onTap: () => widget.onOpenChat(chat.id),
                        onLongPress: () => _showChatMenu(chat),
                        onDelete: () => _confirmDelete(chat),
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }
}

/// A single chat row. Swipe left to surface a delete action (with confirm);
/// long-press to open the rename/star/delete menu.
class _ChatRow extends StatelessWidget {
  const _ChatRow({
    required this.chat,
    required this.onTap,
    required this.onLongPress,
    required this.onDelete,
  });

  final Chat chat;
  final VoidCallback onTap;
  final VoidCallback onLongPress;
  final Future<void> Function() onDelete;

  @override
  Widget build(BuildContext context) {
    final c = ash(context);
    return Dismissible(
      key: ValueKey('chat-${chat.id}'),
      direction: DismissDirection.endToStart,
      // Show the confirm dialog before dismissing so a slip-of-the-thumb
      // doesn't nuke a conversation. Returning false leaves the row in place.
      confirmDismiss: (_) async {
        await onDelete();
        return false;
      },
      background: Container(
        alignment: Alignment.centerRight,
        padding: const EdgeInsets.symmetric(horizontal: 24),
        decoration: BoxDecoration(
          color: Colors.redAccent.withValues(alpha: 0.18),
          borderRadius: BorderRadius.circular(16),
        ),
        child: Icon(Icons.delete_outline, color: Colors.redAccent),
      ),
      child: GestureDetector(
        onTap: onTap,
        onLongPress: onLongPress,
        behavior: HitTestBehavior.opaque,
        child: Glass(
          radius: 16,
          padding: const EdgeInsets.all(12),
          child: Row(
            children: [
              Container(
                width: 40,
                height: 40,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(10),
                  gradient: LinearGradient(
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                    colors: [
                      c.accent.withValues(alpha: 0.5),
                      c.surfaceElev,
                    ],
                  ),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        if (chat.starred) ...[
                          Icon(Icons.star, color: c.accent, size: 14),
                          const SizedBox(width: 4),
                        ],
                        Expanded(
                          child: Text(
                            chat.title,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: TextStyle(
                              color: c.text,
                              fontSize: 14,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 3),
                    Text(
                      chat.preview,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(
                        color: c.textMuted,
                        fontSize: 13,
                        height: 1.3,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              Text(
                chat.when,
                style: TextStyle(
                  color: c.textDim,
                  fontSize: 12,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _MenuRow extends StatelessWidget {
  const _MenuRow({
    required this.icon,
    required this.label,
    required this.iconColor,
    required this.onTap,
    this.labelColor,
  });

  final IconData icon;
  final String label;
  final Color iconColor;
  final Color? labelColor;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final c = ash(context);
    return InkWell(
      onTap: onTap,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        child: Row(
          children: [
            Icon(icon, color: iconColor, size: 20),
            const SizedBox(width: 14),
            Text(
              label,
              style: TextStyle(
                color: labelColor ?? c.text,
                fontSize: 15,
                fontWeight: FontWeight.w500,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Empty-state body for a fresh install. Suggests realistic survival /
/// assistant questions the user could ask. Tapping a chip opens a new
/// chat with that prompt already in the composer — one extra tap from
/// answer.
class _EmptyStatePrompts extends StatelessWidget {
  const _EmptyStatePrompts({required this.c, required this.onPick});

  final AshColors c;
  final ValueChanged<String> onPick;

  static const _prompts = <(IconData, String, String)>[
    (Icons.healing_outlined, 'Cuts and bleeding',
        'I have a deep cut on my arm and it won\'t stop bleeding. What do I do?'),
    (Icons.local_fire_department_outlined, 'Build a fire in the rain',
        'How do I start a fire when everything around me is wet?'),
    (Icons.water_drop_outlined, 'Find or purify water',
        'I\'m running out of water and there\'s no clean source nearby. What are my options?'),
    (Icons.thermostat_outlined, 'Hypothermia signs',
        'How do I tell if someone is getting hypothermia, and what should I do?'),
  ];

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.fromLTRB(20, 24, 20, 120),
      children: [
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 4),
          child: Text(
            'TRY ASKING',
            style: TextStyle(
              color: c.textDim,
              fontSize: 11,
              fontWeight: FontWeight.w600,
              letterSpacing: 0.8,
            ),
          ),
        ),
        const SizedBox(height: 10),
        for (final p in _prompts) ...[
          _PromptChip(
            c: c,
            icon: p.$1,
            title: p.$2,
            prompt: p.$3,
            onTap: () => onPick(p.$3),
          ),
          const SizedBox(height: 8),
        ],
        const SizedBox(height: 4),
        Center(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: Text(
              'Or tap "New conversation" above to start from scratch.',
              textAlign: TextAlign.center,
              style: TextStyle(color: c.textDim, fontSize: 12, height: 1.4),
            ),
          ),
        ),
      ],
    );
  }
}

class _PromptChip extends StatelessWidget {
  const _PromptChip({
    required this.c,
    required this.icon,
    required this.title,
    required this.prompt,
    required this.onTap,
  });

  final AshColors c;
  final IconData icon;
  final String title;
  final String prompt;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        borderRadius: BorderRadius.circular(14),
        onTap: onTap,
        child: Glass(
          radius: 14,
          padding: const EdgeInsets.fromLTRB(12, 12, 12, 12),
          child: Row(
            children: [
              Container(
                width: 36,
                height: 36,
                decoration: BoxDecoration(
                  color: c.accent.withValues(alpha: 0.14),
                  borderRadius: BorderRadius.circular(10),
                ),
                alignment: Alignment.center,
                child: Icon(icon, color: c.accent, size: 18),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      title,
                      style: TextStyle(
                        color: c.text,
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      prompt,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(
                        color: c.textDim,
                        fontSize: 12,
                        height: 1.35,
                      ),
                    ),
                  ],
                ),
              ),
              Icon(Icons.arrow_forward_ios, color: c.textDim, size: 12),
            ],
          ),
        ),
      ),
    );
  }
}
