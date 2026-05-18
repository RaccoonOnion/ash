import 'dart:convert';
import 'dart:math';
import 'dart:ui';

/// Tiered curation borrowed from project-nomad's content model. Lets the
/// Store group packs by depth: tiny first-aid essentials at the top, rare
/// disaster references at the bottom.
enum PackTier {
  essential, // Top-of-bag survival: first aid, fire, water, shelter, navigation.
  standard, // Common emergency scenarios most users will want.
  comprehensive, // Lower-probability or specialty references.
}

extension PackTierExt on PackTier {
  String get label => switch (this) {
        PackTier.essential => 'Essential',
        PackTier.standard => 'Standard',
        PackTier.comprehensive => 'Comprehensive',
      };

  String get blurb => switch (this) {
        PackTier.essential =>
          'Top-of-bag survival — start here.',
        PackTier.standard => 'Common emergencies and injuries.',
        PackTier.comprehensive =>
          'Rare or specialty references.',
      };
}

/// Where a pack ships from. `bundle` packs live inside the app binary
/// (rootBundle); `remote` packs are downloaded over HTTP at install time.
/// Today every pack is `bundle` — the `remote` branch is wired through the
/// service so when packs move to a CDN, no UI rewrite is needed.
enum PackSource { bundle, remote }

class Pack {
  const Pack({
    required this.id,
    required this.name,
    required this.glyph,
    required this.artFrom,
    required this.artTo,
    required this.summary,
    required this.size,
    required this.bytes,
    required this.version,
    required this.entries,
    required this.author,
    required this.tier,
    required this.source,
    this.downloadUrl,
    this.sha256,
    this.installed = false,
  });

  final String id;
  final String name;
  final String glyph;
  final Color artFrom;
  final Color artTo;
  final String summary;
  final String size;
  final double bytes;
  final String version;
  final int entries;
  final String author;
  final PackTier tier;
  final PackSource source;
  final String? downloadUrl;
  final String? sha256;
  final bool installed;

  Pack copyWith({bool? installed}) => Pack(
        id: id,
        name: name,
        glyph: glyph,
        artFrom: artFrom,
        artTo: artTo,
        summary: summary,
        size: size,
        bytes: bytes,
        version: version,
        entries: entries,
        author: author,
        tier: tier,
        source: source,
        downloadUrl: downloadUrl,
        sha256: sha256,
        installed: installed ?? this.installed,
      );

  /// Create a Pack from a registry entry (parsed from packs_registry.json).
  factory Pack.fromRegistry(Map<String, dynamic> m) {
    final packId = m['packId'] as String;
    final packName = m['packName'] as String;
    final chunkCount = m['chunkCount'] as int;
    final fileBytes = (m['fileSizeBytes'] as int?) ?? 0;
    final sizeStr =
        fileBytes > 0 ? '${(fileBytes / 1024).toStringAsFixed(0)} KB' : '? KB';
    final colors = _colorForPack(packId);
    // Tier + source come from the registry when present, otherwise the
    // hardcoded classification (we haven't gone back and edited the registry
    // JSON yet). Keeps the same Pack shape regardless.
    final tierStr = (m['tier'] as String?)?.toLowerCase();
    final tier = _tierForPackId(packId, override: tierStr);
    final sourceStr = (m['source'] as String?)?.toLowerCase();
    final source =
        sourceStr == 'remote' ? PackSource.remote : PackSource.bundle;
    return Pack(
      id: packId,
      name: packName,
      glyph: packName.toUpperCase(),
      artFrom: colors.$1,
      artTo: colors.$2,
      summary: '$packName emergency preparedness and response guide.',
      size: sizeStr,
      bytes: fileBytes.toDouble(),
      version: m['version'] as String? ?? '1.0',
      entries: chunkCount,
      author: 'HazAdapt',
      tier: tier,
      source: source,
      downloadUrl: m['downloadUrl'] as String?,
      sha256: m['sha256'] as String?,
    );
  }
}

/// Hand-classified tier per packId. The registry doesn't ship a `tier` field
/// yet, so we do the classification here. Order: essential covers the
/// "always carry it in your head" survival fundamentals; comprehensive
/// covers low-probability / specialized references; everything else lands in
/// standard. Move to the registry once we touch the JSON anyway.
const _essentialPackIds = <String>{
  'bleeding',
  'burn-injury',
  'choking',
  'cpr',
  'emergency-items',
  'fire',
  'wound',
};

const _comprehensivePackIds = <String>{
  'cicada-brood',
  'dam-and-levee-failure',
  'human-stampede',
  'nuclear-bomb',
  'oil-spills-and-pipeline-leaks',
  'sexually-transmitted-infection',
  'volcano',
};

PackTier _tierForPackId(String packId, {String? override}) {
  if (override == 'essential') return PackTier.essential;
  if (override == 'comprehensive') return PackTier.comprehensive;
  if (override == 'standard') return PackTier.standard;
  if (_essentialPackIds.contains(packId)) return PackTier.essential;
  if (_comprehensivePackIds.contains(packId)) return PackTier.comprehensive;
  return PackTier.standard;
}

/// Deterministic color pair based on pack id hash, for consistent art colors.
(Color, Color) _colorForPack(String id) {
  final hash = id.hashCode;
  final r = Random(hash);
  final from = Color((0xFF << 24) |
      (r.nextInt(128) + 64 << 16) |
      (r.nextInt(128) + 64 << 8) |
      (r.nextInt(128) + 64));
  final to = Color((0xFF << 24) |
      (r.nextInt(64) + 16 << 16) |
      (r.nextInt(64) + 16 << 8) |
      (r.nextInt(64) + 16));
  return (from, to);
}

/// Format byte count as a human-readable string (KB / MB / GB).
String formatBytes(double bytes) {
  if (bytes >= 1024 * 1024 * 1024)
    return '${(bytes / (1024 * 1024 * 1024)).toStringAsFixed(1)} GB';
  if (bytes >= 1024 * 1024)
    return '${(bytes / (1024 * 1024)).toStringAsFixed(1)} MB';
  if (bytes >= 1024) return '${(bytes / 1024).toStringAsFixed(1)} KB';
  return '${bytes.toStringAsFixed(0)} B';
}

/// Load packs from the bundled packs_registry.json asset string.
List<Pack> packsFromRegistry(String jsonStr) {
  final List<dynamic> items = jsonDecode(jsonStr);
  return items
      .map((e) => Pack.fromRegistry(e as Map<String, dynamic>))
      .toList();
}
