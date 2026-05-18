/// One iOS system TTS voice the user can select. Surface for the picker UI.
/// `quality` is `enhanced` / `premium` / `default`; the iPhone usually has a
/// few of each per locale, where Premium voices sound much closer to a real
/// human than the default.
class VoiceOption {
  const VoiceOption({
    required this.name,
    required this.locale,
    required this.quality,
    required this.gender,
  });

  /// Unique identifier from `flutter_tts.getVoices()` — e.g. "Samantha",
  /// "Ava (Enhanced)". Passed back to `setVoice` verbatim.
  final String name;

  /// BCP-47 locale, e.g. "en-US".
  final String locale;

  /// Apple voice quality bucket — one of "premium", "enhanced", "default".
  /// Empty string when the platform didn't surface a value.
  final String quality;

  /// Voice gender if the platform reports it: "male", "female", or empty.
  final String gender;
}

/// Abstract voice I/O contract. The concrete implementation pairs Apple
/// `SFSpeechRecognizer` (via the `speech_to_text` plugin) for streaming STT
/// with Apple `AVSpeechSynthesizer` (via `flutter_tts`) for TTS — no model
/// download, native quality, and partial transcripts as the user talks.
abstract interface class VoiceService {
  /// Whether the platform STT/TTS are available and the recorder has
  /// microphone + speech-recognition permission.
  Future<VoiceStatus> getStatus();

  /// Begin a STT session. Partial transcripts arrive on [partialStream]; the
  /// final string is returned by [stopAndTranscribe]. Calling [startRecording]
  /// when already listening is a no-op.
  Future<void> startRecording();

  /// Stop the recorder and return the final recognized text.
  Future<String> stopAndTranscribe();

  /// Cancel an in-progress recording without producing a final transcript.
  Future<void> cancelRecording();

  bool get isRecording;

  /// Live partials while [isRecording] is true. Emits the rolling transcript
  /// (not just the new word) so the UI can show "what we heard so far". Empty
  /// strings are possible at the very start of the session.
  Stream<String> get partialStream;

  /// Emits the final transcript when STT auto-stops (silence timeout). Live
  /// voice mode uses this to know "user finished talking, send to model".
  Stream<String> get finalStream;

  /// Emits human-readable error strings from the STT engine ("network", "no
  /// speech detected", etc). Lets the live-mode UI surface what went wrong
  /// instead of staring at a frozen "Listening…" pill.
  Stream<String> get errorStream;

  /// Push a chunk of model output into the TTS queue. The service buffers
  /// across calls, detects sentence boundaries (`. ! ? \n`), and speaks each
  /// completed sentence as soon as it's ready. Use [flushTts] to force any
  /// remaining partial sentence to be spoken (e.g. at end of stream).
  Future<void> feedTtsChunk(String chunk);

  /// Speak any text still buffered in [feedTtsChunk] without waiting for a
  /// sentence boundary. Idempotent.
  Future<void> flushTts();

  /// One-shot synthesize, used by the old voice sheet's "speak this message"
  /// path. Bypasses the streaming queue.
  Future<void> speak(String text);

  /// Stop any in-progress TTS playback AND drain the queue. Safe to call
  /// when not speaking.
  Future<void> stopSpeaking();

  bool get isSpeaking;

  /// Emits whenever [isSpeaking] flips. UI uses this to update status pills
  /// ("Speaking…" → "Listening…") and to re-arm the mic for the next turn.
  Stream<bool> get speakingStream;

  /// List the system TTS voices for the current locale, sorted with the
  /// highest-quality variants first. Empty when the platform hasn't
  /// finished initializing.
  Future<List<VoiceOption>> getAvailableVoices();

  /// The voice currently configured. Null when no preference is set
  /// (platform default applies).
  VoiceOption? get currentVoice;

  /// Switch to [voice] and persist the choice so the next launch picks it
  /// up automatically.
  Future<void> setVoice(VoiceOption voice);

  /// Current speech rate, 0.3..0.7. 0.5 is the natural pace.
  double get speechRate;

  /// Change speech rate. Persisted.
  Future<void> setSpeechRate(double rate);

  /// How long the listener tolerates silence before considering the
  /// user's turn done. Clamped 3..10 seconds. The live voice screen's
  /// stuck-mic watchdog scales itself off this value.
  Duration get listeningPatience;

  /// Change listening patience. Persisted.
  Future<void> setListeningPatience(Duration value);

  /// Speak [text] as a one-shot preview. Bypasses the streaming queue and
  /// uses [voice]'s settings (or current voice if [voice] is null). Used
  /// by the picker UI to play a sample sentence when the user taps a row.
  Future<void> previewVoice(VoiceOption voice, String text);

  /// Release recorder, TTS, and platform resources.
  Future<void> dispose();
}

class VoiceStatus {
  const VoiceStatus({
    required this.sttReady,
    required this.ttsReady,
    required this.micPermitted,
    this.error,
  });

  final bool sttReady;
  final bool ttsReady;
  final bool micPermitted;
  final String? error;
}
