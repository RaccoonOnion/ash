import 'package:flutter/material.dart';
import 'package:flutter_gemma/flutter_gemma.dart';
import 'app.dart';
import 'services/apple_voice_service.dart';
import 'services/gemma_inference_service.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  const huggingFaceToken = String.fromEnvironment('HUGGINGFACE_TOKEN');
  await FlutterGemma.initialize(
    huggingFaceToken: huggingFaceToken.isEmpty ? null : huggingFaceToken,
    webStorageMode: WebStorageMode.cacheApi,
  );
  runApp(
    AshApp(
      service: GemmaInferenceService(),
      voiceService: AppleVoiceService(),
    ),
  );
}
