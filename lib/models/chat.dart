import 'dart:typed_data';

import '../services/inference_service.dart';
import '../services/inference_settings.dart';

class ChatMessage {
  const ChatMessage({
    required this.id,
    required this.role,
    required this.text,
    this.imageBytes,
    this.streaming = false,
    this.sources,
  });
  final int id;
  final String role; // 'user' | 'assistant'
  final String text;
  final Uint8List? imageBytes;
  final bool streaming;

  /// Reference chunks that grounded this assistant reply. Set on a preflight
  /// chunk from the inference service before token streaming starts; rendered
  /// as citation chips under the assistant bubble. Null for user messages
  /// and for assistant replies that didn't use retrieval.
  final List<MessageSource>? sources;
}

class Chat {
  Chat({
    required this.id,
    required this.title,
    required this.when,
    required this.preview,
    this.lensPackIds,
    this.starred = false,
    this.initialComposerText,
    List<ChatMessage>? messages,
    InferenceSettings? inferenceSettings,
  })  : messages = messages ?? [],
        inferenceSettings = inferenceSettings ?? InferenceSettings.defaults;

  /// One-shot composer prefill. Set when a chat is created from an
  /// example-prompt chip on the home screen; the chat screen reads it
  /// in initState, pushes it into the composer, then clears the field
  /// so a re-mount doesn't re-prefill. Never persisted.
  String? initialComposerText;

  final String id;
  String title;
  String when;
  String preview;
  bool starred;

  /// Which knowledge packs this chat retrieves from. `null` means "all
  /// installed packs" (the default). A specific [Set] scopes retrieval to
  /// just those packs — used e.g. when "Ask Ash about this" opens a chat
  /// from a single Library section, or when the user taps the lens pill
  /// in the header and narrows the scope.
  ///
  /// Mutated in place when the user changes the lens. Listed last in the
  /// constructor since most call sites omit it.
  Set<String>? lensPackIds;

  final List<ChatMessage> messages;
  // Inference knobs for this conversation. Mutated in place when the user
  // tweaks them from the per-chat settings sheet — re-assigning the whole
  // Chat would lose `messages` continuity and force the model session to
  // rebuild on every slider drag.
  InferenceSettings inferenceSettings;
}
