import AVFoundation
import Flutter
import UIKit

@main
@objc class AppDelegate: FlutterAppDelegate, FlutterImplicitEngineDelegate {
  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  func didInitializeImplicitFlutterEngine(_ engineBridge: FlutterImplicitEngineBridge) {
    GeneratedPluginRegistrant.register(with: engineBridge.pluginRegistry)

    // Audio session control channel. flutter_tts on iOS only calls
    // AVAudioSession.setActive(false) when an utterance finishes naturally
    // (didFinish delegate). When TTS is cancelled mid-utterance — which is
    // what live-mode interrupt does — the session stays active in
    // playAndRecord mode. After a few interrupt cycles the session state
    // corrupts AVAudioEngine.inputNode, causing speech_to_text's next
    // listenForSpeech to SIGSEGV inside AVAudioNode outputFormatForBus:.
    // Dart calls this channel after stopSpeaking to force a clean release.
    let audioChannel = FlutterMethodChannel(
      name: "ash/audio_session",
      binaryMessenger: engineBridge.applicationRegistrar.messenger())
    audioChannel.setMethodCallHandler { (call, result) in
      switch call.method {
      case "deactivate":
        do {
          try AVAudioSession.sharedInstance().setActive(
            false, options: .notifyOthersOnDeactivation)
          result(true)
        } catch {
          // Swift error (not NSException). Safe to surface to Dart.
          result(FlutterError(
            code: "deactivate_failed",
            message: error.localizedDescription, details: nil))
        }
      default:
        result(FlutterMethodNotImplemented)
      }
    }
  }
}
