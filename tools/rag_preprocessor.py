#!/usr/bin/env python3
"""RAG Pre-processing Pipeline.

Reads markdown files from a source directory, chunks each by ## headings,
embeds each chunk using the same ONNX MiniLM model the app uses, and writes
one JSON file per source plus a packs_registry.json manifest.

Usage:
    python3 tools/rag_preprocessor.py \
        --source ../raw-data-markdown \
        --output assets/rag/packs \
        --model assets/models/minilm.onnx \
        --vocab assets/models/vocab.txt
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


# ---------------------------------------------------------------------------
# BERT WordPiece tokenizer (mirrors the Dart BertTokenizer exactly)
# ---------------------------------------------------------------------------

class BertTokenizer:
    def __init__(self, vocab_path: str):
        self.vocab: dict[str, int] = {}
        with open(vocab_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                token = line.rstrip("\n")
                self.vocab[token] = idx
        self.unk_id = self.vocab.get("[UNK]", 100)
        self.cls_id = self.vocab.get("[CLS]", 101)
        self.sep_id = self.vocab.get("[SEP]", 102)
        # Collect subword prefixes: tokens starting with ## (but not ## itself)
        # In BERT, continuation tokens start with "##"
        self._basic_pattern = re.compile(
            r"[^\W\d_]+|\d+|[^\s\w]"  # letters | digits | punctuation
        )

    def _basic_tokenize(self, text: str) -> list[str]:
        text = text.lower().strip()
        # Insert spaces around punctuation
        text = re.sub(r"([^\s\w])", r" \1 ", text)
        tokens = text.split()
        return [t for t in tokens if t]

    def _wordpiece(self, word: str, max_subwords: int = 100) -> list[str]:
        if word in self.vocab:
            return [word]
        tokens: list[str] = []
        start = 0
        while start < len(word) and len(tokens) < max_subwords:
            end = len(word)
            found = False
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = "##" + substr
                if substr in self.vocab:
                    tokens.append(substr)
                    found = True
                    break
                end -= 1
            if not found:
                tokens.append("[UNK]")
                break
            start = end
        return tokens

    def tokenize(self, text: str, max_length: int = 128) -> tuple[list[int], list[int]]:
        """Returns (input_ids, attention_mask)."""
        basic = self._basic_tokenize(text)
        subwords: list[str] = []
        for word in basic:
            subwords.extend(self._wordpiece(word))

        # Truncate to max_length - 2 (for [CLS] and [SEP])
        subwords = subwords[: max_length - 2]

        input_ids = [self.cls_id] + [self.vocab.get(t, self.unk_id) for t in subwords] + [self.sep_id]
        attention_mask = [1] * len(input_ids)

        # Pad to max_length
        pad_len = max_length - len(input_ids)
        input_ids += [0] * pad_len
        attention_mask += [0] * pad_len

        return input_ids, attention_mask


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def create_embedder(model_path: str) -> ort.InferenceSession:
    return ort.InferenceSession(model_path)


def embed_text(
    session: ort.InferenceSession,
    tokenizer: BertTokenizer,
    text: str,
    max_length: int = 128,
) -> list[float]:
    input_ids, attention_mask = tokenizer.tokenize(text, max_length)

    import numpy as np

    inputs = {
        "input_ids": np.array([input_ids], dtype=np.int64),
        "attention_mask": np.array([attention_mask], dtype=np.int64),
        "token_type_ids": np.array([np.zeros(max_length, dtype=np.int64)], dtype=np.int64),
    }

    outputs = session.run(None, inputs)
    # Output shape: [1, seq_len, 384]
    hidden = outputs[0][0]  # [seq_len, 384]

    # Mean pooling over non-padded tokens
    mask = np.array(attention_mask)
    valid = mask.sum()
    if valid > 0:
        embedding = hidden[:valid].mean(axis=0)
    else:
        embedding = hidden[0]

    # L2 normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding.tolist()


# ---------------------------------------------------------------------------
# Markdown chunker
# ---------------------------------------------------------------------------

def slugify(name: str) -> str:
    """Convert a filename like 'car-accident.md' to a pack id like 'car-accident'."""
    return Path(name).stem.lower().replace(" ", "-").replace("&", "and")


def title_from_filename(filename: str) -> str:
    """Derive a display title from the filename."""
    stem = Path(filename).stem
    # Replace hyphens with spaces, title-case words
    return stem.replace("-", " ").replace("&", "&").title()


def chunk_markdown(text: str, source_slug: str) -> list[dict]:
    """Split markdown into chunks at ## boundaries.

    Returns list of {chunkId, text, sectionPath}.
    """
    lines = text.split("\n")
    chunks: list[dict] = []

    # Extract h1 title (first # line)
    h1_title = None
    h1_end = 0
    for i, line in enumerate(lines):
        if line.startswith("# "):
            h1_title = line[2:].strip()
            h1_end = i + 1
            break

    # Skip the blockquote after h1 if present
    content_start = h1_end
    for i in range(h1_end, len(lines)):
        if lines[i].startswith("> "):
            content_start = i + 1
        elif lines[i].strip() == "":
            continue
        else:
            content_start = i
            break

    # Find all ## headings and split into sections
    sections: list[tuple[str, int, int]] = []  # (heading, start_line, end_line)
    current_heading = "__intro__"
    current_start = content_start

    for i in range(content_start, len(lines)):
        line = lines[i]
        if line.startswith("## "):
            # Close previous section
            if current_start < i:
                sections.append((current_heading, current_start, i))
            current_heading = line[3:].strip()
            current_start = i + 1

    # Close last section
    if current_start < len(lines):
        sections.append((current_heading, current_start, len(lines)))

    # Convert sections to chunks
    for heading, start, end in sections:
        section_text = "\n".join(lines[start:end]).strip()
        if not section_text:
            continue

        # If section is very long (>2000 chars), split at ### boundaries
        if len(section_text) > 2000:
            sub_chunks = _split_at_h3(section_text, source_slug, heading)
            chunks.extend(sub_chunks)
        else:
            section_slug = heading.lower().replace(" ", "-").replace("(", "").replace(")", "")
            chunk_id = f"{source_slug}-{section_slug}"
            chunks.append({
                "chunkId": chunk_id,
                "text": section_text,
                "sectionPath": heading if heading != "__intro__" else "",
            })

    # Filter out the __intro__ sectionPath
    for c in chunks:
        if c["sectionPath"] == "":
            # Try to use h1 title or first meaningful content
            pass

    return chunks


def _split_at_h3(text: str, source_slug: str, parent_heading: str) -> list[dict]:
    """Split a long section at ### boundaries."""
    lines = text.split("\n")
    sub_sections: list[tuple[str, int, int]] = []
    current_heading = "__top__"
    current_start = 0

    for i, line in enumerate(lines):
        if line.startswith("### "):
            if current_start < i:
                sub_sections.append((current_heading, current_start, i))
            current_heading = line[4:].strip()
            current_start = i + 1

    if current_start < len(lines):
        sub_sections.append((current_heading, current_start, len(lines)))

    chunks = []
    for heading, start, end in sub_sections:
        section_text = "\n".join(lines[start:end]).strip()
        if not section_text:
            continue
        section_slug = heading.lower().replace(" ", "-").replace("(", "").replace(")", "")
        chunk_id = f"{source_slug}-{parent_heading.lower().replace(' ', '-').replace('(', '').replace(')', '')}-{section_slug}"
        chunks.append({
            "chunkId": chunk_id,
            "text": section_text,
            "sectionPath": f"{parent_heading} > {heading}" if heading != "__top__" else parent_heading,
        })

    return chunks


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_file(
    filepath: str,
    session: ort.InferenceSession,
    tokenizer: BertTokenizer,
) -> tuple[dict, dict]:
    """Process a single markdown file. Returns (pack_json, registry_entry)."""

    filename = os.path.basename(filepath)
    pack_id = slugify(filename)
    pack_name = title_from_filename(filename)

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    raw_chunks = chunk_markdown(text, pack_id)

    # Embed each chunk
    chunks = []
    for chunk in raw_chunks:
        embedding = embed_text(session, tokenizer, chunk["text"])
        chunks.append({
            "chunkId": chunk["chunkId"],
            "text": chunk["text"],
            "source": pack_id,
            "sectionPath": chunk["sectionPath"],
            "sourceType": "hazards",
            "embeddingModelVersion": "all-MiniLM-L6-v2",
            "embedding": embedding,
        })

    pack_data = {
        "packId": pack_id,
        "packName": pack_name,
        "version": "1.0",
        "embeddingModel": "all-MiniLM-L6-v2",
        "sourceType": "hazards",
        "chunks": chunks,
    }

    registry_entry = {
        "packId": pack_id,
        "packName": pack_name,
        "version": "1.0",
        "chunkCount": len(chunks),
        "sourceFile": filename,
    }

    return pack_data, registry_entry


def main():
    parser = argparse.ArgumentParser(description="RAG Pre-processing Pipeline")
    parser.add_argument(
        "--source", default="../raw-data-markdown",
        help="Directory containing .md source files"
    )
    parser.add_argument(
        "--output", default="assets/rag/packs",
        help="Output directory for pack JSONs"
    )
    parser.add_argument(
        "--model", default="assets/models/minilm.onnx",
        help="Path to ONNX embedding model"
    )
    parser.add_argument(
        "--vocab", default="assets/models/vocab.txt",
        help="Path to BERT vocab file"
    )
    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    source_dir = (project_root / args.source).resolve()
    output_dir = (project_root / args.output).resolve()
    model_path = (project_root / args.model).resolve()
    vocab_path = (project_root / args.vocab).resolve()

    print(f"Source:  {source_dir}")
    print(f"Output:  {output_dir}")
    print(f"Model:   {model_path}")
    print(f"Vocab:   {vocab_path}")

    # Validate inputs
    if not source_dir.is_dir():
        print(f"Error: source directory not found: {source_dir}", file=sys.stderr)
        sys.exit(1)
    if not model_path.exists():
        print(f"Error: model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    if not vocab_path.exists():
        print(f"Error: vocab file not found: {vocab_path}", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize embedder
    print("Loading ONNX model...")
    session = create_embedder(str(model_path))
    tokenizer = BertTokenizer(str(vocab_path))
    print("Model loaded.")

    # Process each markdown file
    md_files = sorted(source_dir.glob("*.md"))
    if not md_files:
        print(f"No .md files found in {source_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(md_files)} markdown files.")

    registry = []
    for md_file in md_files:
        print(f"  Processing {md_file.name}...", end=" ", flush=True)
        pack_data, registry_entry = process_file(str(md_file), session, tokenizer)

        # Write pack JSON
        out_path = output_dir / f"{pack_data['packId']}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(pack_data, f, ensure_ascii=False)

        file_size = out_path.stat().st_size
        registry_entry["fileSizeBytes"] = file_size
        registry.append(registry_entry)
        print(f"{len(pack_data['chunks'])} chunks, {file_size:,} bytes")

    # Write registry
    registry_path = output_dir / "packs_registry.json"
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(registry)} packs written to {output_dir}")
    print(f"Registry: {registry_path}")


if __name__ == "__main__":
    main()
