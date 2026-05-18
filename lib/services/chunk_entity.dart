import 'package:objectbox/objectbox.dart';

@Entity()
class ChunkEntity {
  @Id()
  int id;

  @Index()
  String chunkId;

  String text;

  @Index()
  String source;

  String? sectionPath;

  @Index()
  String sourceType;

  @Index()
  String packId;

  String? dlcId;
  String? docId;
  String? embeddingModelVersion;

  @HnswIndex(
    dimensions: 384,
    distanceType: VectorDistanceType.cosine,
  )
  @Property(type: PropertyType.floatVector)
  List<double>? embedding;

  ChunkEntity({
    this.id = 0,
    this.chunkId = '',
    this.text = '',
    this.source = '',
    this.sectionPath,
    this.sourceType = 'core',
    this.packId = '',
    this.dlcId,
    this.docId,
    this.embeddingModelVersion,
    this.embedding,
  });
}
