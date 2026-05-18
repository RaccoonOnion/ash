/// The on-device LLM variants Ash supports. Each variant resolves to a single
/// `.litertlm` file on HuggingFace under the `litert-community` org. Only one
/// model is *active* (loaded into the engine) at a time — the SDK supports
/// having multiple files installed on disk and switching the active one via
/// [GemmaInferenceService.setActiveLlmModel].
enum LlmModel {
  /// Gemma 4 E2B-it — 2B-effective parameters. Recommended default. Smaller
  /// download, faster first token, lower thermal load. Good fit for
  /// iPhone 15/16 and excellent on iPhone 17+.
  gemma4E2B(
    filename: 'gemma-4-E2B-it.litertlm',
    hfRepo: 'litert-community/gemma-4-E2B-it-litert-lm',
    displayName: 'Gemma 4 E2B',
    paramLabel: 'E2B',
    sizeGB: 1.4,
    etaMinutes: 12,
    blurb: 'Balanced quality and speed. Runs fully on-device with RAG.',
  ),

  /// Gemma 4 E4B-it — 4B-effective parameters. Higher reasoning quality but
  /// a much larger download (~3.7 GB) and noticeably slower first token /
  /// hotter sustained use. Best on iPhone 17 Pro+. Below 6 GB of free RAM
  /// the engine may OOM during vision swaps.
  gemma4E4B(
    filename: 'gemma-4-E4B-it.litertlm',
    hfRepo: 'litert-community/gemma-4-E4B-it-litert-lm',
    displayName: 'Gemma 4 E4B',
    paramLabel: 'E4B',
    sizeGB: 3.7,
    etaMinutes: 30,
    blurb: 'Higher quality reasoning. Best on iPhone 17 Pro+. Larger download.',
  );

  const LlmModel({
    required this.filename,
    required this.hfRepo,
    required this.displayName,
    required this.paramLabel,
    required this.sizeGB,
    required this.etaMinutes,
    required this.blurb,
  });

  /// Local filename `FlutterGemma.installModel` uses to identify the model on
  /// disk. Also the key returned by `listInstalledModels`.
  final String filename;

  /// HuggingFace `<user>/<repo>` slug where the `.litertlm` blob lives.
  final String hfRepo;

  /// Human-readable name shown in pickers and settings.
  final String displayName;

  /// Short parameter-class badge (e.g. "E2B"). Used in the model card UI.
  final String paramLabel;

  /// Approximate download size in gigabytes. Surfaces in the model picker so
  /// the user can plan disk + cellular use.
  final double sizeGB;

  /// Rough first-time download ETA on a typical home Wi-Fi connection.
  /// Drives the progress banner's countdown. Not authoritative — actual time
  /// is recomputed against measured throughput.
  final int etaMinutes;

  /// One-liner description shown on the model card.
  final String blurb;

  /// Fully-qualified resolve URL for `FlutterGemma.installModel(...).fromNetwork(...)`.
  String get url => 'https://huggingface.co/$hfRepo/resolve/main/$filename';

  /// Lookup by the legacy [ModelInfo.id] string used in
  /// [model_pick_screen.dart] so the picker can drive activation without
  /// every screen needing to import this enum directly.
  static LlmModel? fromId(String id) {
    for (final m in LlmModel.values) {
      if (m.filename.startsWith(id) || id == m.filename.replaceAll('.litertlm', '')) {
        return m;
      }
    }
    return null;
  }
}
