# RAG Evaluation

Offline retrieval-quality benchmark for the app's RAG pipeline. It replicates the
Dart hybrid retrieval logic (BM25 + vector search + RRF fusion) in pure Python so
you can measure retrieval quality against a synthetic query set — no app build, no
device, no LLM-in-the-loop scoring.

The flow is three steps:

1. **Configure** an LLM endpoint (used only to generate the benchmark queries).
2. **Generate** a benchmark dataset of synthetic emergency queries from the packs.
3. **Evaluate** retrieval against that dataset and read the metrics.

## Contents

| File | Role |
|------|------|
| `generate_dataset.py` | Calls an LLM to synthesize 2–3 realistic queries per pack chunk → `benchmark/dataset.json` |
| `run_eval.py`         | Runs retrieval for every query, computes metrics, writes timestamped results |
| `retrieval.py`        | Pure-Python `RetrievalEngine`: BM25 + ONNX vector search + RRF fusion |
| `metrics.py`          | MRR, NDCG@k, Hit Rate@k, and RAGAS ID-based precision/recall |

## Setup

```bash
pip install -r tools/rag_eval/requirements.txt
```

`retrieval.py` imports `BertTokenizer`, `create_embedder`, and `embed_text` from
`tools/rag_preprocessor.py` (one directory up), so run the commands from the
project root and keep that file in place.

## Step 1 — Configure the env

Query generation talks to an Anthropic-compatible API. Copy the example and fill
it in:

```bash
cp tools/rag_eval/.env.example tools/rag_eval/.env
```

`.env` is loaded automatically (looked up next to the script, then in the project
root). Existing environment variables always take precedence. Supported keys:

| Variable | Purpose | Default |
|----------|---------|---------|
| `LLM_API_TOKEN`  | API key (falls back to `ANTHROPIC_API_KEY`) | — (required) |
| `LLM_BASE_URL`   | Anthropic-compatible base URL, e.g. `https://api.z.ai/api/anthropic` | Anthropic default |
| `LLM_MODEL`      | Model id (e.g. `glm-4.6` for z.ai) | `claude-sonnet-4-20250514` |
| `LLM_TIMEOUT_MS` | Request timeout in milliseconds | `3000000` |

Only generation needs the API. **Step 3 (`run_eval.py`) is fully offline** and
needs no API key.

## Step 2 — Generate the dataset

```bash
python tools/rag_eval/generate_dataset.py \
    --packs assets/rag/packs \
    --output benchmark/dataset.json
```

- Loads every chunk from the pack JSONs (skips `packs_registry.json`).
- Generates up to 3 queries per chunk, each tagged with its gold `chunk_id`.
- Saves after every batch and **resumes** from an existing `dataset.json`, so an
  interrupted run can be re-run safely.

## Step 3 — Run the evaluation

```bash
python tools/rag_eval/run_eval.py \
    --dataset benchmark/dataset.json \
    --packs assets/rag/packs \
    --model assets/models/minilm.onnx \
    --vocab assets/models/vocab.txt
```

Prints overall + per-pack metrics and writes a timestamped JSON to
`benchmark/results/<timestamp>.json`.

### Useful flags

| Flag | Default | Notes |
|------|---------|-------|
| `--mode` | `hybrid` | `hybrid` (RRF), `vector`, or `bm25` |
| `--top-k` | `10` | Final results returned per query |
| `--vector-top-k` | `5` | Vector candidates before fusion |
| `--bm25-top-k` | `10` | BM25 candidates before fusion |
| `--rrf-k` | `60` | RRF fusion constant |
| `--filter-pack` | — | Restrict to one pack (quick smoke test) |

Quick smoke test on a single pack:

```bash
python tools/rag_eval/run_eval.py \
    --dataset benchmark/dataset.json \
    --packs assets/rag/packs \
    --model assets/models/minilm.onnx \
    --vocab assets/models/vocab.txt \
    --filter-pack bleeding
```

## Reading the metrics

The dataset has exactly **one gold chunk per query**, which mechanically caps some
metrics — `recall@k == hit_rate@k`, and set-based precision is bounded by `1/k`.
`run_eval.py` prints a metric guide after each run that interprets the numbers
against those ceilings.

| Metric | Range | Meaning |
|--------|-------|---------|
| MRR | 0–1 | Avg `1/rank` of the gold chunk; `1.0` = always ranked #1 |
| NDCG@k | 0–1 | Position-discounted ranking quality in the top-k |
| Hit Rate@k (= Recall@k) | 0–1 | Fraction of queries with the gold chunk in top-k |
| RAGAS Precision | 0–`1/k` | `|retrieved ∩ relevant| / |retrieved|`; low value is structural dilution, not a bad retriever |
| RAGAS Recall | 0–1 | `|retrieved ∩ relevant| / |relevant|` — "did we retrieve the answer at all" |
