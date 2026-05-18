#!/usr/bin/env python3
"""Offline re-chunker for Ash's bundled HazAdapt knowledge packs.

The original packs were chunked at the coarsest possible boundary —
`(packId, phase, audience)` — producing chunks of up to 16 KB that bundled
3–6 self-contained sub-topics into one atomic embedding. Retrieval against
those chunks is noisy: a query like "where do I hide during an active
shooter" matches the whole `RUN.HIDE.FIGHT.` chunk but the injected slice
ends up wherever the markdown happens to start (usually RUN).

This script splits each chunk at internal `####` headings, merges tiny
fragments back into their neighbors, and re-embeds every resulting
sub-chunk with the same `sentence-transformers/all-MiniLM-L6-v2` model the
Flutter app uses at query time. The pack JSON format is preserved so the
existing import path picks up the new data unchanged.

Usage:
    cd <repo root>
    source tools/.venv/bin/activate
    python tools/rechunk.py

This rewrites every `assets/rag/packs/*.json` in place and updates
`packs_registry.json` with new chunk counts + file sizes. The Flutter
app needs a one-shot DB wipe after running this (see the round-4
migration marker in `lib/app.dart`) so the freshly-embedded chunks
replace the old ones in ObjectBox.
"""
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import List, Tuple

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ─── Paths ───────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent
PACKS_DIR = REPO_ROOT / "assets" / "rag" / "packs"
REGISTRY = PACKS_DIR / "packs_registry.json"

# ─── Tuning ──────────────────────────────────────────────────────────────────

# Chunks shorter than this get merged with the next block. A 100-char chunk
# is usually just a heading with no body — useless for retrieval and a
# magnet for false-positive matches.
MIN_CHARS = 200

# Soft target for max chunk size after splitting. We don't split further if
# a sub-block exceeds this; markdown structure inside is the model's
# problem to handle.
SOFT_MAX_CHARS = 1500

# The bundled model the Flutter app loads from `assets/models/minilm.onnx`.
# Must match — otherwise embeddings live in a different space and the query-
# time MiniLM run produces vectors that don't align with what's in DB.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Bumped when the chunking pipeline changes. Stored on every chunk so the
# Flutter side can detect stale on-device data and re-import.
EMBEDDING_MODEL_VERSION = "all-MiniLM-L6-v2-r4"

# ─── Sanitization (mirrors lib/services/chunk_sanitizer.dart) ────────────────

# Codepoints to strip outright: zero-width spaces, bidi marks, BOMs, line
# separators, word joiners. Matches `_shouldDrop` on the Dart side so we
# embed the same text the app shows the user.
_DROP_CODEPOINTS = set()
for cp in range(0x00, 0x20):
    if cp not in (0x09, 0x0A):  # keep \t, \n
        _DROP_CODEPOINTS.add(cp)
_DROP_CODEPOINTS.add(0x7F)
_DROP_CODEPOINTS.update(range(0x200B, 0x2010))
_DROP_CODEPOINTS.update([0x2028, 0x2029])
_DROP_CODEPOINTS.update(range(0x202A, 0x202F))
_DROP_CODEPOINTS.update(range(0x2060, 0x2065))
_DROP_CODEPOINTS.add(0xFEFF)

_SMART_QUOTE_FOLD = {
    "“": '"', "”": '"',
    "‘": "'", "’": "'",
    "—": "-", "–": "-",
    "…": "...",
}

_HTML_TAG = re.compile(r"<[a-zA-Z/][^>]*>")
_FOOTNOTE = re.compile(r"\[\^[^\]]*\]")
_CITE = re.compile(r"\{cite:[^}]*\}")
_MULTI_NL = re.compile(r"\n{3,}")
_MULTI_WS = re.compile(r"[ \t]+")


def sanitize(text: str) -> str:
    """Same scrub the Flutter sanitizer does at injection time. Doing it at
    build time means the embedding matches what we'll show the model."""
    # 1. Drop control / zero-width / bidi codepoints.
    chars = [ch for ch in text if ord(ch) not in _DROP_CODEPOINTS]
    s = "".join(chars)
    # 2. Fold smart quotes / dashes / ellipsis.
    for src, dst in _SMART_QUOTE_FOLD.items():
        s = s.replace(src, dst)
    # 3. Strip HTML, footnotes, citation placeholders.
    s = _HTML_TAG.sub("", s)
    s = _FOOTNOTE.sub("", s)
    s = _CITE.sub("", s)
    # 4. Collapse whitespace.
    s = _MULTI_NL.sub("\n\n", s)
    s = _MULTI_WS.sub(" ", s)
    # 5. Trim trailing whitespace per line.
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    # 6. Dedup consecutive identical non-blank lines, ignoring blank
    #    separators in between AND markdown-heading vs plain-text variants
    #    of the same string. The HazAdapt source frequently has the section
    #    title both as a heading and as a paragraph rephrase right after
    #    (e.g. "#### Avalanche Starting Below You\n\nAvalanche Starting
    #    Below You\n\n- ..."). We normalize each line by stripping leading
    #    `#`s + bold markers before comparing.
    def _key(line: str) -> str:
        return line.strip().lstrip("#").lstrip("*").strip().rstrip("*").strip()

    out_lines = []
    last_key = None
    for line in s.split("\n"):
        norm = line.strip()
        if not norm:
            out_lines.append(line)
            continue
        k = _key(line)
        if k and k == last_key:
            continue  # drop the dup
        out_lines.append(line)
        last_key = k
    s = "\n".join(out_lines).strip()
    return unicodedata.normalize("NFC", s)


# ─── Chunk splitting ─────────────────────────────────────────────────────────

_HEADING_PAT = re.compile(r"^#{2,6}\s+(.+)$")


def split_at_headings(text: str) -> List[Tuple[str, str]]:
    """Split markdown into (heading, body) tuples at `##`+ boundaries.
    The first tuple may have an empty heading if the text starts with body
    text (some chunks lead with a bold paragraph, not a heading)."""
    blocks: List[Tuple[str, str]] = []
    current_heading = ""
    current_body: List[str] = []

    def flush():
        body = "\n".join(current_body).strip()
        if current_heading or body:
            blocks.append((current_heading, body))

    for line in text.split("\n"):
        m = _HEADING_PAT.match(line.strip())
        if m:
            flush()
            current_heading = m.group(1).strip()
            current_body = []
        else:
            current_body.append(line)
    flush()
    return blocks


def merge_small_blocks(blocks: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Glue tiny sub-blocks onto their following neighbor so we don't ship
    100-char "heading only" chunks that match queries on title alone.

    Returns (heading, body) tuples where `body` does NOT include the
    leading heading — the caller renders it via [_render] exactly once.
    When a merged-in neighbor had its own heading, that heading IS inlined
    into the body as a markdown `####` line (it's a real sub-topic that
    didn't earn its own block)."""
    if not blocks:
        return []
    merged: List[Tuple[str, str]] = []
    i = 0
    while i < len(blocks):
        h, body = blocks[i]
        # Size check uses the rendered form so a 60-char body under a
        # 120-char heading isn't endlessly re-merged.
        while len(_render(h, body)) < MIN_CHARS and i + 1 < len(blocks):
            nh, nb = blocks[i + 1]
            neighbor = _render(nh, nb)
            body = f"{body}\n\n{neighbor}".strip() if body else neighbor
            i += 1
        merged.append((h, body))
        i += 1
    return merged


def _render(heading: str, body: str) -> str:
    if heading and body:
        return f"#### {heading}\n\n{body}"
    if heading:
        return f"#### {heading}"
    return body


def _slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9-]+", "-", s.lower()).strip("-")
    return s[:48] or "block"


# ─── Per-pack rebuild ────────────────────────────────────────────────────────


def rebuild_pack_chunks(pack: dict) -> List[dict]:
    """Split + sanitize + assign new chunkIds. Embeddings get filled in by
    the caller in one batched encode (faster than per-chunk on CPU)."""
    new_chunks: List[dict] = []
    pack_id = pack["packId"]
    for parent_idx, orig in enumerate(pack["chunks"]):
        text = sanitize(orig.get("text", ""))
        if not text:
            continue
        # Split at internal headings, then merge tiny fragments.
        blocks = merge_small_blocks(split_at_headings(text))
        if not blocks:
            blocks = [("", text)]
        parent_section = orig.get("sectionPath") or ""
        section_slug = _slug(parent_section)
        for i, (heading, body) in enumerate(blocks):
            chunk_text = _render(heading, body).strip()
            if not chunk_text:
                continue
            heading_slug = _slug(heading) if heading else f"b{i}"
            # Include parent-chunk index so two source chunks that produced
            # same-named sub-blocks (e.g. both having a leading bodyless
            # block "b0") don't collide on chunkId.
            chunk_id = (
                f"{pack_id}-c{parent_idx}-{section_slug}-{heading_slug}-{i}"
            )
            new_chunks.append({
                "chunkId": chunk_id,
                "text": chunk_text,
                "source": orig.get("source", pack_id),
                "sectionPath": orig.get("sectionPath"),
                "sourceType": orig.get("sourceType", "core"),
                "embeddingModelVersion": EMBEDDING_MODEL_VERSION,
            })
    return new_chunks


# ─── Main ────────────────────────────────────────────────────────────────────


def main() -> int:
    if not PACKS_DIR.exists():
        print(f"Packs dir not found: {PACKS_DIR}", file=sys.stderr)
        return 1

    print(f"Loading {MODEL_NAME}…")
    model = SentenceTransformer(MODEL_NAME)
    print(f"  max_seq_length: {model.max_seq_length}")
    print(f"  embedding dim:  {model.get_sentence_embedding_dimension()}")

    pack_files = sorted(
        p for p in PACKS_DIR.glob("*.json") if p.name != "packs_registry.json"
    )
    print(f"\nFound {len(pack_files)} pack files")

    # Stats for reporting.
    orig_total = 0
    new_total = 0
    size_by_id: dict[str, tuple[int, int]] = {}

    for pack_path in tqdm(pack_files, desc="Re-chunking", unit="pack"):
        with pack_path.open() as f:
            pack = json.load(f)
        orig_total += len(pack["chunks"])

        new_chunks = rebuild_pack_chunks(pack)
        new_total += len(new_chunks)

        if new_chunks:
            texts = [c["text"] for c in new_chunks]
            # normalize_embeddings=True matches the runtime path: mean-pool
            # → L2 normalize. Both ends produce comparable vectors.
            vecs = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            for c, v in zip(new_chunks, vecs):
                # Use 6-digit float precision — keeps JSON small without
                # measurably hurting cosine similarity at 384 dimensions.
                c["embedding"] = [round(float(x), 6) for x in v.tolist()]

        pack["chunks"] = new_chunks
        pack["embeddingModel"] = "all-MiniLM-L6-v2"
        pack["embeddingModelVersion"] = EMBEDDING_MODEL_VERSION

        with pack_path.open("w", encoding="utf-8") as f:
            # Compact JSON — no extra whitespace. The bundle's 5MB → ~9MB
            # range is mostly embeddings; pretty-printing would balloon it.
            json.dump(pack, f, ensure_ascii=False, separators=(",", ":"))

        size_by_id[pack["packId"]] = (len(new_chunks), pack_path.stat().st_size)

    # Update registry with new counts + sizes so the Flutter app's
    # `Pack.size` / `Pack.entries` show truth.
    with REGISTRY.open() as f:
        registry = json.load(f)
    for entry in registry:
        pid = entry.get("packId")
        if pid in size_by_id:
            count, size = size_by_id[pid]
            entry["chunkCount"] = count
            entry["fileSizeBytes"] = size
            entry["embeddingModelVersion"] = EMBEDDING_MODEL_VERSION
    with REGISTRY.open("w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)

    print(f"\nDone.")
    print(f"  Original chunks: {orig_total}")
    print(f"  New chunks:      {new_total}  ({new_total / orig_total:.1f}x)")
    print(f"  Embedding model: {EMBEDDING_MODEL_VERSION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
