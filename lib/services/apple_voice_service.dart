import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:speech_to_text/speech_recognition_result.dart';
import 'package:speech_to_text/speech_to_text.dart';

import 'voice_service.dart';

/// Method channel for forcing an AVAudioSession deactivation from Dart.
/// Implemented in AppDelegate.swift. Needed because flutter_tts only
/// deactivates the session on natural utterance completion — cancel/stop
/// leaves the session live, which corrupts AVAudioEngine state on the next
/// listen and SIGSEGVs after a few stop cycles in live mode.
const _audioSessionChannel = MethodChannel('ash/audio_session');

/// VoiceService implementation that wraps Apple's native engines:
///   - SFSpeechRecognizer (via the `speech_to_text` plugin) for STT
///   - AVSpeechSynthesizer (via `flutter_tts`) for TTS
///
/// Replaces the old Whisper-tiny ONNX path — same interface, no model
/// download, native streaming partials. Adds a sentence-chunked TTS queue so
/// live voice mode can feed it model tokens as they stream and have the
/// device read sentence-by-sentence.
class AppleVoiceService implements VoiceService {
  AppleVoiceService();

  final _stt = SpeechToText();
  final _tts = FlutterTts();

  bool _sttInitialized = false;
  bool _ttsInitialized = false;
  bool _disposed = false;
  bool _listening = false;
  bool _speaking = false;
  String _lastTranscript = '';

  // Voice preference state, loaded from NSUserDefaults on first init.
  VoiceOption? _currentVoice;
  double _speechRate = 0.5;
  Duration _listeningPatience = const Duration(seconds: 5);
  // One-shot guard so the post-init auto-pick runs at most once per
  // process lifetime. Without this, every getStatus / speak / listen
  // call would re-run the voice scan.
  bool _voiceAutoPickAttempted = false;
  static const _kPrefVoiceName = 'tts_voice_name';
  static const _kPrefVoiceLocale = 'tts_voice_locale';
  static const _kPrefVoiceQuality = 'tts_voice_quality';
  static const _kPrefVoiceGender = 'tts_voice_gender';
  static const _kPrefSpeechRate = 'tts_speech_rate';
  static const _kPrefListeningPatience = 'stt_listening_patience_s';
  // One-shot migration marker. Some early testers had the slider
  // saved at 10s (the previous max) — we now want everyone back on
  // the 5s default; users who care about a longer window can slide
  // up again. Bumping this string nukes the prior value once more.
  static const _kPrefPatienceDefaultMigration = 'stt_patience_default_5_v1';

  final _partialController = StreamController<String>.broadcast();
  final _finalController = StreamController<String>.broadcast();
  final _speakingController = StreamController<bool>.broadcast();
  final _errorController = StreamController<String>.broadcast();

  // Buffer of model tokens fed to [feedTtsChunk] that haven't yet hit a
  // sentence boundary. Once we see `. ! ? \n` we slice off the sentence,
  // enqueue it for speech, and keep the remainder.
  final _ttsBuffer = StringBuffer();
  // FIFO of sentences ready to speak. flutter_tts is async/single-utterance,
  // so we drain serially.
  final _ttsQueue = <String>[];
  bool _ttsDraining = false;

  // Sentence-end characters. We treat newline as a boundary too so list
  // items and code-block lines get spoken individually.
  static final _boundaryRegex = RegExp(r'[.!?\n]');

  Future<void> _ensureInit() async {
    if (_sttInitialized && _ttsInitialized) return;
    if (!_sttInitialized) {
      try {
        _sttInitialized = await _stt.initialize(
          onError: (e) {
            debugPrint('[voice] STT error: $e');
            _errorController.add(e.errorMsg);
          },
          onStatus: (s) {
            debugPrint('[voice] STT status: $s');
            if (s == SpeechToText.notListeningStatus ||
                s == SpeechToText.doneStatus) {
              _listening = false;
            }
          },
          debugLogging: false,
        );
      } catch (e) {
        debugPrint('[voice] STT initialize failed: $e');
        _errorController.add('STT init failed: $e');
      }
    }
    if (!_ttsInitialized) {
      try {
        // CRITICAL on iOS: explicitly configure AVAudioSession so:
        //  - TTS playback is routed to the SPEAKER (default routes to the
        //    earpiece after a record session → user hears nothing),
        //  - the session can be reactivated for a follow-up STT turn (without
        //    this, the second listen() call after a TTS playback silently
        //    fails — the symptom user reports as "second round listen
        //    doesn't work").
        await _tts.setSharedInstance(true);
        await _tts.setIosAudioCategory(
          IosTextToSpeechAudioCategory.playAndRecord,
          [
            IosTextToSpeechAudioCategoryOptions.defaultToSpeaker,
            IosTextToSpeechAudioCategoryOptions.allowBluetooth,
            IosTextToSpeechAudioCategoryOptions.mixWithOthers,
          ],
          IosTextToSpeechAudioMode.voiceChat,
        );
        await _tts.setLanguage('en-US');
        await _tts.setPitch(1.0);
        // Pull persisted preferences (set last session) and apply before
        // the first speak(). If nothing's saved we leave the defaults.
        await _loadAndApplyPrefs();
        _tts.setStartHandler(() {
          _speaking = true;
          _speakingController.add(true);
        });
        _tts.setCompletionHandler(() {
          _speaking = false;
          _speakingController.add(false);
          _drainTtsQueue(); // pull next sentence if any
        });
        _tts.setCancelHandler(() {
          _speaking = false;
          _speakingController.add(false);
        });
        _tts.setErrorHandler((e) {
          debugPrint('[voice] TTS error: $e');
          _speaking = false;
          _speakingController.add(false);
        });
        _ttsInitialized = true;
      } catch (e) {
        debugPrint('[voice] TTS initialize failed: $e');
      }
    }
    // Auto-pick best voice — fire AFTER _ttsInitialized is true so the
    // nested _ensureInit call inside getAvailableVoices short-circuits.
    // Unawaited because the caller (often live-voice _start) doesn't
    // need the voice picked before the mic comes up; the next
    // utterance just uses whatever voice the picker lands on by the
    // time it speaks. Guarded so the scan only runs once.
    if (_ttsInitialized && !_voiceAutoPickAttempted) {
      _voiceAutoPickAttempted = true;
      unawaited(_autoPickBestVoice());
    }
  }

  Future<void> _autoPickBestVoice() async {
    try {
      final picked = await _pickBestVoice();
      if (picked == null) return;
      _currentVoice = picked;
      try {
        await _tts.setVoice({'name': picked.name, 'locale': picked.locale});
      } catch (e) {
        debugPrint('[voice] auto-pick setVoice failed: $e');
        _currentVoice = null;
      }
    } catch (e) {
      debugPrint('[voice] auto-pick scan failed: $e');
    }
  }

  @override
  Future<VoiceStatus> getStatus() async {
    await _ensureInit();
    final micOk = await _stt.hasPermission;
    return VoiceStatus(
      sttReady: _sttInitialized,
      ttsReady: _ttsInitialized,
      micPermitted: micOk,
    );
  }

  // --- STT ---

  @override
  bool get isRecording => _listening;

  @override
  Stream<String> get partialStream => _partialController.stream;

  @override
  Stream<String> get finalStream => _finalController.stream;

  @override
  Future<void> startRecording() async {
    if (_listening) return;
    await _ensureInit();
    if (!_sttInitialized) {
      throw StateError('Speech recognition not available');
    }
    _lastTranscript = '';
    _listening = true;
    try {
      await _stt.listen(
        onResult: _onSttResult,
        listenOptions: SpeechListenOptions(
          partialResults: true,
          cancelOnError: true,
          // We DON'T force onDevice: true here. On iOS the on-device
          // language pack only works if the user has it downloaded for the
          // active locale, and SFSpeechRecognizer silently fails if it's
          // missing. Default (false) lets iOS pick — uses on-device when
          // available, network when not. For airplane-mode survival use,
          // pre-download via Settings → General → Language → Add Language
          // on Wi-Fi, then iOS will use on-device automatically.
          onDevice: false,
        ),
        // pauseFor = silence window after which SFSpeechRecognizer fires
        // finalResult and ends the turn. Configurable via Settings →
        // Listening patience. Higher = more tolerant of mid-sentence
        // pauses, at the cost of a longer latency floor on short replies.
        // Independent of TTS speechRate: that controls how fast the model
        // reads its reply aloud, this controls how patient the listener
        // is with the user.
        pauseFor: _listeningPatience,
      );
    } catch (e) {
      _listening = false;
      debugPrint('[voice] listen() threw: $e');
      _errorController.add('Listen failed: $e');
      rethrow;
    }
  }

  void _onSttResult(SpeechRecognitionResult result) {
    _lastTranscript = result.recognizedWords;
    if (_disposed) return;
    if (result.finalResult) {
      _listening = false;
      _finalController.add(_lastTranscript);
    } else {
      _partialController.add(_lastTranscript);
    }
  }

  @override
  Future<String> stopAndTranscribe() async {
    if (!_listening) return _lastTranscript;
    await _stt.stop();
    _listening = false;
    // speech_to_text fires final result via _onSttResult shortly after stop().
    // Give it a moment so the final-text-with-confidence variant lands.
    await Future<void>.delayed(const Duration(milliseconds: 150));
    return _lastTranscript;
  }

  @override
  Future<void> cancelRecording() async {
    if (!_listening) return;
    await _stt.cancel();
    _listening = false;
    _lastTranscript = '';
  }

  // --- TTS ---

  @override
  // Intentionally NOT checking `_ttsDraining` — that flag races against the
  // iOS completion handler. On iOS, `_tts.speak()`'s Future resolves when
  // the utterance is queued, not when it finishes; the completion handler
  // fires after speech actually ends and sets `_speaking = false`. If we
  // included `_ttsDraining` here, `isSpeaking` could return true briefly
  // after the handler clears `_speaking` and before the `.then((_) =>
  // _ttsDraining = false)` callback runs — exactly the window when live
  // mode's `_onSpeakingChanged(false)` tries to re-arm the mic, but bails
  // because we mis-reported "still speaking." That's the "stops listening
  // after first reply" symptom.
  bool get isSpeaking => _speaking || _ttsQueue.isNotEmpty;

  @override
  Stream<bool> get speakingStream => _speakingController.stream;

  @override
  Future<void> speak(String text) async {
    if (text.trim().isEmpty) return;
    await _ensureInit();
    if (!_ttsInitialized) return;
    // One-shot speak bypasses the streaming queue: stop anything in-flight,
    // clear the queue, then speak this string as a single utterance.
    await stopSpeaking();
    _speaking = true;
    _speakingController.add(true);
    await _tts.speak(text);
  }

  @override
  Future<void> feedTtsChunk(String chunk) async {
    if (chunk.isEmpty) return;
    await _ensureInit();
    if (!_ttsInitialized) return;
    _ttsBuffer.write(chunk);

    // Slice off every complete sentence in the buffer and enqueue it. Leave
    // the remainder (potential partial sentence) for the next chunk.
    var text = _ttsBuffer.toString();
    while (true) {
      final match = _boundaryRegex.firstMatch(text);
      if (match == null) break;
      final endIdx = match.end;
      final sentence = text.substring(0, endIdx).trim();
      text = text.substring(endIdx);
      if (sentence.isNotEmpty) _ttsQueue.add(sentence);
    }
    _ttsBuffer
      ..clear()
      ..write(text);

    _drainTtsQueue();
  }

  @override
  Future<void> flushTts() async {
    final remainder = _ttsBuffer.toString().trim();
    _ttsBuffer.clear();
    if (remainder.isNotEmpty) _ttsQueue.add(remainder);
    _drainTtsQueue();
  }

  void _drainTtsQueue() {
    if (_ttsDraining || _speaking) return;
    if (_ttsQueue.isEmpty) return;
    _ttsDraining = true;
    final next = _ttsQueue.removeAt(0);
    // Fire and forget — completion handler will trigger the next drain.
    _tts.speak(next).then((_) {
      _ttsDraining = false;
    }).catchError((Object e) {
      debugPrint('[voice] TTS speak error: $e');
      _ttsDraining = false;
      _drainTtsQueue();
    });
  }

  @override
  Future<void> stopSpeaking() async {
    _ttsBuffer.clear();
    _ttsQueue.clear();
    _ttsDraining = false;
    if (_ttsInitialized) {
      try {
        await _tts.stop();
      } catch (_) {}
    }
    // CRITICAL on iOS: flutter_tts only calls AVAudioSession.setActive(false)
    // when an utterance finishes naturally via didFinish. When we stop
    // mid-utterance (which is what live-mode interrupt does), the
    // session stays active in playAndRecord. After 3-4 stop cycles the
    // session state corrupts AVAudioEngine.inputNode, and the next
    // STT listen() segfaults inside AVAudioNode outputFormatForBus:
    // (faulting thread on AVAudioIOUnit queue, null deref in
    // GetClientFormat). Forcing a deactivation here clears the leaked
    // state. Best-effort — channel may fail if AppDelegate hasn't wired
    // up the handler yet (early-startup race), and the OS will simply
    // refuse if the session was already inactive.
    if (Platform.isIOS) {
      try {
        await _audioSessionChannel.invokeMethod<bool>('deactivate');
      } catch (e) {
        debugPrint('[apple-voice] audio session deactivate failed: $e');
      }
    }
    _speaking = false;
    _speakingController.add(false);
  }

  // --- Voice selection ---

  @override
  VoiceOption? get currentVoice => _currentVoice;

  @override
  double get speechRate => _speechRate;

  @override
  Future<List<VoiceOption>> getAvailableVoices() async {
    await _ensureInit();
    if (!_ttsInitialized) return const [];
    try {
      final raw = await _tts.getVoices;
      if (raw is! List) return const [];

      // Strict: en-* AND (premium OR enhanced) only. Default-quality
      // voices (robotic) are never returned — they're not useful to
      // anyone and the picker UI has been removed anyway. Prefer en-US
      // before other English variants. Dedupe by name.
      final shortlist = <VoiceOption>[];
      final seenNames = <String>{};
      for (final v in raw) {
        if (v is! Map) continue;
        final name = (v['name'] as String?) ?? '';
        final locale = (v['locale'] as String?) ?? '';
        if (name.isEmpty) continue;
        if (!locale.toLowerCase().startsWith('en')) continue;
        final quality = (v['quality'] as String?)?.toLowerCase() ?? '';
        if (quality != 'enhanced' && quality != 'premium') continue;
        if (!seenNames.add(name)) continue;
        shortlist.add(VoiceOption(
          name: name,
          locale: locale,
          quality: quality,
          gender: (v['gender'] as String?)?.toLowerCase() ?? '',
        ));
      }

      // Sort: premium → enhanced; en-US before other en-*; ties by name.
      int qRank(String q) => q == 'premium' ? 0 : 1;
      int lRank(String l) => l.toLowerCase() == 'en-us' ? 0 : 1;
      shortlist.sort((a, b) {
        final byQ = qRank(a.quality).compareTo(qRank(b.quality));
        if (byQ != 0) return byQ;
        final byL = lRank(a.locale).compareTo(lRank(b.locale));
        if (byL != 0) return byL;
        return a.name.compareTo(b.name);
      });
      return shortlist;
    } catch (e) {
      debugPrint('[voice] getVoices failed: $e');
      return const [];
    }
  }

  /// Auto-pick the single best installed voice. Returns null if no
  /// Premium/Enhanced en-* voice is installed — caller leaves the
  /// platform default in place rather than picking a robotic one.
  Future<VoiceOption?> _pickBestVoice() async {
    final voices = await getAvailableVoices();
    if (voices.isEmpty) return null;
    // getAvailableVoices already sorts premium-en-US first, so the
    // head of the list is the best choice.
    return voices.first;
  }

  @override
  Future<void> setVoice(VoiceOption voice) async {
    await _ensureInit();
    if (!_ttsInitialized) return;
    try {
      await _tts.setVoice({'name': voice.name, 'locale': voice.locale});
      _currentVoice = voice;
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString(_kPrefVoiceName, voice.name);
      await prefs.setString(_kPrefVoiceLocale, voice.locale);
      await prefs.setString(_kPrefVoiceQuality, voice.quality);
      await prefs.setString(_kPrefVoiceGender, voice.gender);
    } catch (e) {
      debugPrint('[voice] setVoice failed: $e');
    }
  }

  @override
  Future<void> setSpeechRate(double rate) async {
    final clamped = rate.clamp(0.3, 0.7);
    _speechRate = clamped;
    try {
      await _tts.setSpeechRate(clamped);
      final prefs = await SharedPreferences.getInstance();
      await prefs.setDouble(_kPrefSpeechRate, clamped);
    } catch (e) {
      debugPrint('[voice] setSpeechRate failed: $e');
    }
  }

  @override
  Duration get listeningPatience => _listeningPatience;

  @override
  Future<void> setListeningPatience(Duration value) async {
    final seconds = value.inSeconds.clamp(3, 10);
    _listeningPatience = Duration(seconds: seconds);
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setInt(_kPrefListeningPatience, seconds);
    } catch (e) {
      debugPrint('[voice] setListeningPatience failed: $e');
    }
    // Applied to the NEXT listen() — the current recognizer session keeps
    // running with whatever pauseFor it was started with. No need to
    // interrupt the in-flight turn just to change the silence threshold.
  }

  @override
  Future<void> previewVoice(VoiceOption voice, String text) async {
    await _ensureInit();
    if (!_ttsInitialized) return;
    // Stop anything in flight + clear the queue so the preview lands cleanly.
    await stopSpeaking();
    try {
      // Temporarily switch to the preview voice. Don't persist — the user
      // hasn't committed to it yet. If they pick it, setVoice() will run.
      await _tts.setVoice({'name': voice.name, 'locale': voice.locale});
      await _tts.speak(text);
      // After the preview ends, restore the persisted voice (the
      // completion handler will eventually fire, but we proactively set
      // it back so the next streaming reply sounds right). If
      // [_currentVoice] is null, leave the temporary voice — first
      // launch, nothing committed yet.
      if (_currentVoice != null) {
        // Schedule the restore for after the preview utterance finishes —
        // doing it synchronously would override the in-flight speak().
        unawaited(_restoreVoiceAfterPreview());
      }
    } catch (e) {
      debugPrint('[voice] previewVoice failed: $e');
    }
  }

  Future<void> _restoreVoiceAfterPreview() async {
    // Poll until TTS reports idle, then snap back to the persisted voice.
    while (_speaking) {
      await Future<void>.delayed(const Duration(milliseconds: 100));
    }
    final v = _currentVoice;
    if (v != null) {
      try {
        await _tts.setVoice({'name': v.name, 'locale': v.locale});
      } catch (_) {}
    }
  }

  Future<void> _loadAndApplyPrefs() async {
    // IMPORTANT: this runs from inside _ensureInit BEFORE _ttsInitialized
    // is set. It MUST NOT call anything that triggers a nested
    // _ensureInit (getAvailableVoices, etc.) — that creates an infinite
    // recursion that hangs the voice service on first use. Voice
    // auto-pick is deferred until _autoPickBestVoice() fires at the end
    // of _ensureInit, when the flag is already true.
    try {
      final prefs = await SharedPreferences.getInstance();
      final rate = prefs.getDouble(_kPrefSpeechRate);
      _speechRate = (rate ?? 0.5).clamp(0.3, 0.7);
      await _tts.setSpeechRate(_speechRate);
      // One-time wipe of stale persisted patience values from early
      // testing so the slider snaps back to the 5s default. Runs once
      // per device per migration marker.
      final migrated =
          prefs.getBool(_kPrefPatienceDefaultMigration) ?? false;
      if (!migrated) {
        await prefs.remove(_kPrefListeningPatience);
        await prefs.setBool(_kPrefPatienceDefaultMigration, true);
      }
      final patience = prefs.getInt(_kPrefListeningPatience);
      _listeningPatience = Duration(seconds: (patience ?? 5).clamp(3, 10));
    } catch (e) {
      // Prefs not available yet — fall back to defaults already set.
      debugPrint('[voice] _loadAndApplyPrefs: $e');
      _speechRate = 0.5;
      await _tts.setSpeechRate(_speechRate);
      _listeningPatience = const Duration(seconds: 5);
    }
  }

  // --- Lifecycle ---

  @override
  Future<void> dispose() async {
    _disposed = true;
    await cancelRecording();
    await stopSpeaking();
    await _partialController.close();
    await _finalController.close();
    await _speakingController.close();
    await _errorController.close();
  }

  @override
  Stream<String> get errorStream => _errorController.stream;
}
