# Ash — Project Highlights

> Resume-style technical summary of Ash, an offline survival assistant for iOS.

---

## 1. Project Overview

**Ash** is an offline-first survival assistant iOS app that runs Google's Gemma 4 AI models **fully on-device** — no cloud dependency after initial model download. Users can ask emergency-response questions via text, camera, or voice and receive grounded, citation-linked answers even in airplane mode.

- **Hackathon**: Built for the [Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon) (Kaggle × Google DeepMind, May 2026)
- **Team**: [Yunxiang Yan](https://github.com/RaccoonOnion) and [Yao Xiao](https://github.com/yaoxiao6)
- **Platform**: iOS 17+ (iPhone 15 Pro or newer) — built with Flutter 3.6+
- **Key stats**:
  - **56 emergency-response knowledge packs** covering CPR, severe bleeding, flash floods, hypothermia, active shooter, overdose, nuclear fallout, and more
  - **2 model sizes**: Gemma 4 E2B (1.4 GB, default) and Gemma 4 E4B (3.7 GB, stronger reasoning) — hot-swappable at runtime
  - **Fully offline** after initial ~1.4 GB model download over Wi-Fi

---

## 2. Technical Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Flutter UI Layer                          │
│  ┌─────────────┐  ┌──────────────────┐  ┌─────────────────────┐ │
│  │  Composer    │  │ Live Voice Screen│  │ Library Reader      │ │
│  │ text·camera·mic│ │ orb·captions·TTS│  │ citation deep-link  │ │
│  └──────┬───────┘  └───────┬──────────┘  └─────────────────────┘ │
│         │                  │                                     │
│         ▼                  ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              GemmaInferenceService                          │ │
│  │    Hybrid RAG · BM25+vector · RRF · 2-pass rewrite         │ │
│  └──────┬──────────────────┬──────────────────┬───────────────┘ │
│         │                  │                  │                  │
└─────────┼──────────────────┼──────────────────┼──────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
   ┌──────────────┐  ┌──────────────┐   ┌──────────────────────┐
   │ MiniLM-L6-v2 │  │ ObjectBox    │   │ Gemma 4 E2B / E4B    │
   │ 384-dim embed │  │ HNSW index   │   │ via LiteRT-LM        │
   │ ONNX Runtime │  │ cosine dist  │   │ + MTP spec. decode   │
   └──────┬───────┘  └──────────────┘   │ text·vision dual-eng │
          │                              └──────────┬───────────┘
   ┌──────┴───────┐                                 │
   │ BM25 Search  │                                 │
   │ inverted idx │                                 │
   │ k1=1.5 b=.75 │                                 │
   └──────┬───────┘                                 │
          │                                         │
   ┌──────┴───────┐                                 │
   │  RRF Fusion  │                                 │
   │  k=60        │                                 │
   └──────────────┘──── context ──────────────────┘
                                                    │
                                          ┌─────────┴─────────┐
                                          ▼                   ▼
                                   Markdown bubbles    AVSpeechSynthesizer
                                   + citation chips    (streaming TTS)
```

### Core Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **UI** | Flutter + Dart | Cross-platform UI with native iOS integration |
| **LLM Runtime** | LiteRT-LM (Google AI Edge) | On-device inference for `.litertlm` model files |
| **Models** | Gemma 4 E2B-it / E4B-it | Primary text + vision inference |
| **Embedding** | MiniLM-L6-v2 (ONNX Runtime) | 384-dim sentence embeddings for dense RAG path |
| **Vector DB** | ObjectBox with HNSW | On-device cosine-distance vector search (dense path) |
| **Sparse Retrieval** | BM25 (hand-written in Dart) | Keyword search with inverted index (sparse path) |
| **Fusion** | Reciprocal Rank Fusion (RRF, k=60) | Merges dense + sparse ranked lists into final retrieval |
| **Speech-to-Text** | Apple SFSpeechRecognizer | On-device streaming transcription |
| **Text-to-Speech** | Apple AVSpeechSynthesizer | Sentence-level streaming synthesis |
| **GPU Backend** | Metal delegate | Hardware-accelerated inference with CPU fallback |

---

## 3. Key Technical Highlights

### 3.1 Full On-Device LLM Inference

No cloud dependency after model download. The Gemma 4 models (up to 3.7 GB) run entirely on the iPhone's Neural Engine / GPU via Google's LiteRT-LM runtime. All inference — text generation, vision encoding, RAG retrieval, embedding — happens locally. Airplane-mode safe.

**Key files**: `lib/services/gemma_inference_service.dart`, `lib/services/llm_model.dart`

### 3.2 Hybrid RAG with BM25 + Vector Retrieval Fused via RRF

The retrieval pipeline combines **dense vector search** and **sparse keyword search** into a single ranked list using Reciprocal Rank Fusion (RRF), then optionally applies an adaptive 2-pass rewrite.

**Dense (vector) path** — MiniLM-L6-v2 embeds the user query into a 384-dim vector; ObjectBox HNSW returns the 5 nearest chunks by cosine distance, threshold-filtered at 0.65 to reject sentence-structure matches that aren't topically relevant.

**Sparse (BM25) path** — A hand-written `Bm25Search` class implements the Robertson–Sparck Jones BM25 scoring formula (k1=1.5, b=0.75) with an inverted index built lazily from all ObjectBox chunks. Tokenization strips punctuation, lowercases, and removes ~30 English stop words. BM25 independently retrieves its top 10 keyword-matched candidates.

**Fusion** — The two ranked lists are merged with Reciprocal Rank Fusion: each chunk's final score is `sum(1/(k + rank))` summed across both lists (k=60, the standard default from Cormack, Clarke & Büttcher 2009). Chunks that appear in both lists get a boosted score; chunks unique to one list still contribute. The fused list is re-sorted by RRF score descending.

**Adaptive 2-pass rewrite** — If the best cosine distance from the vector pass is above a threshold (0.5, indicating the query was too vague or colloquial), the LLM rewrites the query into a search-ready question and a second hybrid retrieval runs. The system keeps whichever pass produced the better vector score. This rewrite is disabled in live voice mode for latency stability.

The hybrid approach catches cases where pure semantic search misses exact keyword matches (e.g., "tourniquet" matches the BM25 index precisely but may be overshadowed in embedding space by broader wound-care concepts), while the vector path handles paraphrases and conceptual matches that keyword search cannot.

**Key files**: `lib/services/bm25_search.dart` (BM25 scorer), `lib/services/gemma_inference_service.dart` (`_retrieveHybrid()` at ~1822, `_fuseWithRrf()` at ~1755, `query()` at ~692)

### 3.3 Speculative Decoding via Multi-Token Prediction (MTP)

MTP drafter models are bundled inside the `.litertlm` blobs alongside the base Gemma 4 weights. When speculative decoding is enabled (`enableSpeculativeDecoding: true`), the drafter predicts multiple candidate tokens in parallel, and the base model verifies them in a single forward pass. This yields approximately **1.5–2x decode speedup** with no quality loss.

**Key file**: `lib/services/gemma_inference_service.dart` (`_speculativeDecoding` flag)

### 3.4 Custom BERT WordPiece Tokenizer (Hand-Written in Dart)

The MiniLM-L6-v2 embedding model requires BERT-style WordPiece tokenization, but no production-quality Dart implementation existed. We wrote one from scratch:

- **Basic tokenizer**: Lowercasing, punctuation splitting, whitespace tokenization
- **WordPiece subword tokenizer**: Greedy longest-match against the 30,522-token BERT vocabulary
- Handles `[CLS]`/`[SEP]` special tokens, truncation to 128 tokens, and attention mask generation
- Loads `vocab.txt` (BERT WordPiece vocabulary) from bundled assets

**Key file**: `lib/services/bert_tokenizer.dart`

### 3.5 Streaming TTS Pipeline with Sentence-Boundary Chunking

Model output streams token-by-token, but AVSpeechSynthesizer needs complete sentences for natural prosody. The voice service solves this with:

- A **rolling buffer** that accumulates streamed tokens until a sentence boundary is detected (`[.!?\n]` regex)
- A **FIFO queue** of complete sentences drained serially by the synthesizer
- Text sanitization via `sanitizeForTts()` that strips markdown, citation chips, and formatting before synthesis

This creates the perception of real-time voice: the model streams a response and the first sentence starts playing aloud within ~1 second of generation, while subsequent sentences queue behind it.

**Key files**: `lib/services/apple_voice_service.dart` (`feedTtsChunk()`), `lib/services/tts_sanitizer.dart`

### 3.6 Context Window Management

Gemma 4 supports up to 32k token context windows, but on a mobile device the KV cache grows proportionally. The app tracks projected KV-cache utilization and when it crosses **85%**, surfaces a warning banner with two one-tap actions:

- **Extend**: Bump `maxTokens` up (triggers an engine reload, ~5s for text-only / ~30s for vision-capable)
- **Trim**: Drop the oldest 30% of conversation turns and reset the chat session, replaying remaining history so the model stays coherent

The context estimator calculates projected cache size from the current token count plus the estimated tokens from the incoming query and retrieved RAG chunks (~3,000 character budget for context injection).

**Key file**: `lib/services/context_estimator.dart`

### 3.7 EOS Marker Detection with Rolling Buffer

LiteRT-LM streams tokens as subword pieces, so the end-of-turn marker `<end_of_turn>` can arrive fragmented across multiple inference chunks (e.g., `<end`, `_of`, `_turn`, `>`). A rolling buffer holds back ambiguous tokens and checks for stop markers (`<end_of_turn>`, `<eos>`, `<start_of_turn>`) before emitting. This prevents the model from "drifting" into pretraining noise or multilingual gibberish after a response should have ended.

**Key file**: `lib/services/gemma_inference_service.dart` (`emitSafe()` method, lines ~810–859)

### 3.8 Chunk Sanitization Pipeline

Raw knowledge-base markdown chunks contain artifacts that confuse the LLM or produce poor RAG results: Unicode curiosities, smart quotes, HTML tags, Pandoc footnotes, citation placeholders, and duplicate lines from source extraction. A multi-stage pipeline cleans chunks before they reach the model:

1. **Unicode cleanup**: Strip invisible codepoints, normalize smart quotes and dashes, fold ellipsis
2. **HTML stripping**: Remove tags, Pandoc footnotes (`[^1]`), citation placeholders (`[citation:N]`)
3. **Deduplication**: Remove consecutive identical lines, collapse excessive whitespace

**Key file**: `lib/services/chunk_sanitizer.dart` (`sanitizeChunkText()`, lines ~376–440)

### 3.9 Sub-Block Relevance Scoring

Retrieved RAG chunks can be large (multiple markdown sections). Rather than injecting the full chunk into the prompt, the system splits at `####` headings and scores each sub-block using **stopword-filtered term-overlap** with the user's query. Only the top-scoring sub-blocks (up to the character budget) are included in the context. This gives the model precisely relevant material instead of entire pages of loosely related content.

**Key file**: `lib/services/chunk_sanitizer.dart` (`selectRelevantContent()`, lines ~237–289)

### 3.10 Engine Lifecycle Management

The app manages **two engine configurations**: a lightweight text-only engine (fast startup, low memory) and a vision-capable engine (~30s cold load, higher memory). Key behaviors:

- **Sticky vision**: Once the vision engine is activated (first image query), it remains loaded for subsequent queries — no repeated 30s waits
- **GPU-to-CPU fallback**: If the Metal delegate crashes (SIGSEGV on engine rebuild, observed in TestFlight builds), subsequent engine rebuilds use CPU-only inference. A `_gpuRebuildUnsafe` flag tracks this state.
- **Session persistence**: Chat sessions maintain a KV cache across turns; the session rebuilds only when chat ID or sampling parameters change, automatically replaying conversation history

**Key file**: `lib/services/gemma_inference_service.dart`

### 3.11 Voice Service — Native Apple Speech Integration

Ash integrates Apple's native speech frameworks through a combination of Flutter plugins and custom method channels:

- **STT**: SFSpeechRecognizer via `speech_to_text` plugin — on-device when the language pack is installed, streaming partial transcripts with configurable listening patience (3–10 seconds)
- **TTS**: AVSpeechSynthesizer via `flutter_tts` — sentence-queued streaming, premium/enhanced voice auto-selection, persistent voice preferences
- **Audio session management**: Custom `ash/audio_session` method channel in `AppDelegate.swift` for forced AVAudioSession deactivation after TTS stop, preventing session corruption that causes SIGSEGV in subsequent STT sessions
- **Category**: `playAndRecord` with default-to-speaker routing to support simultaneous microphone input and audio output

**Key files**: `lib/services/apple_voice_service.dart`, `ios/Runner/AppDelegate.swift`

---

## 4. Notable Engineering Challenges Solved

### iOS Jetsam SIGKILL on Vision Encoder
The Gemma 4 vision encoder requires significantly more memory than the text-only path. iOS's Jetsam watchdog silently killed the process mid-load until we added `Extended Virtual Addressing` and `Increased Memory Limit` entitlements to `Runner.entitlements`. Without a paid Apple Developer team, these entitlements are unavailable and vision mode cannot load.

### Metal Delegate SIGSEGV on Engine Rebuild
After closing and rebuilding the inference engine (e.g., when switching models or changing `maxTokens`), the Metal GPU delegate would occasionally segfault. We implemented a `_gpuRebuildUnsafe` flag that tracks whether a GPU delegate failure has occurred and falls back to CPU-only inference for all subsequent rebuilds. The initial engine load still attempts Metal.

### AVAudioSession Corruption After Repeated STT/TTS Stop Cycles
Rapidly toggling between speech recognition and text-to-speech — the core loop of live voice mode — would occasionally corrupt the AVAudioSession state. The next STT attempt would fail with a SIGSEGV or silently return no results. Fixed by adding a custom method channel (`ash/audio_session`) in `AppDelegate.swift` that forces `AVAudioSession.sharedInstance().setActive(false)` after TTS stop, ensuring a clean session for the next STT activation.

### App Store Rejection 90208 (MinimumOSVersion Mismatch)
App Store Connect rejected the IPA with error 90208 because some bundled CocoaPods frameworks declared a `MinimumOSVersion` in their `Info.plist` that didn't match the app's deployment target. We wrote `fix_framework_plists.sh` — a post-build script that patches all framework `Info.plist` files to align `MinimumOSVersion` with the app's `IPHONEOS_DEPLOYMENT_TARGET`.

### Fragmented EOS Tokens Causing Multilingual Drift
When `<end_of_turn>` arrived fragmented across subword boundaries (e.g., `<end` + `_of_turn>`), the model would continue generating text — often drifting into other languages or pretraining noise. The `emitSafe()` rolling buffer holds back tokens that might be part of a stop marker, checks for complete marker strings, and truncates the response cleanly.

---

## 5. Tech Stack Summary

| Category | Technologies |
|----------|-------------|
| **Languages** | Dart, Swift, Python |
| **Frameworks** | Flutter 3.6+, LiteRT-LM, ONNX Runtime, ObjectBox |
| **AI/ML Models** | Gemma 4 E2B-it / E4B-it, MiniLM-L6-v2, MTP speculative decoding drafter |
| **iOS Native** | Metal GPU delegate, SFSpeechRecognizer, AVSpeechSynthesizer, AVAudioSession |
| **Vector Search** | ObjectBox HNSW index (cosine distance, 384-dimensional) |
| **Build & Distribution** | CocoaPods, Xcode 15+, TestFlight, GitHub Releases |
| **Model Hosting** | HuggingFace (`litert-community`) |
| **Content** | 56 emergency-response knowledge packs (JSON + markdown) |
| **Tools** | Python preprocessing pipeline (`rag_preprocessor.py`, `embed_propositions.py`), custom `fix_framework_plists.sh` |

---

## 6. Repository

**GitHub**: [github.com/RaccoonOnion/ash](https://github.com/RaccoonOnion/ash)
**License**: Apache 2.0
**TestFlight**: [testflight.apple.com/join/z5vsJM8A](https://testflight.apple.com/join/z5vsJM8A)
