#!/usr/bin/env python3
"""Generate synthetic benchmark queries for RAG evaluation.

For each chunk in the pack JSONs, calls an Anthropic-compatible LLM API to
generate 2-3 realistic emergency queries. Outputs benchmark/dataset.json.

Works with Anthropic directly or any Anthropic-compatible endpoint (e.g.
z.ai's GLM) via these env vars:
    LLM_API_TOKEN  — API key (falls back to ANTHROPIC_API_KEY)
    LLM_BASE_URL   — custom base URL, e.g. https://api.z.ai/api/anthropic
    LLM_MODEL      — model id (default: claude-sonnet-4-20250514)
    LLM_TIMEOUT_MS — request timeout in ms (default: 3000000)

Usage:
    # Anthropic (default)
    export ANTHROPIC_API_KEY=sk-...
    python tools/rag_eval/generate_dataset.py \
        --packs assets/rag/packs \
        --output benchmark/dataset.json

    # z.ai GLM
    export LLM_API_TOKEN=...
    export LLM_BASE_URL=https://api.z.ai/api/anthropic
    export LLM_MODEL=glm-4.6
    export LLM_TIMEOUT_MS=3000000
    python tools/rag_eval/generate_dataset.py
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("Error: 'anthropic' package required. Install with: pip install anthropic", file=sys.stderr)
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Error: 'tqdm' package required. Install with: pip install tqdm", file=sys.stderr)
    sys.exit(1)


def _load_dotenv():
    """Load KEY=VALUE pairs from a .env file into os.environ.

    Looks for .env next to this script, then in the project root. Existing
    environment variables take precedence (we never overwrite them). Uses
    python-dotenv if installed, otherwise a minimal built-in parser so no
    extra dependency is required.
    """
    script_dir = Path(__file__).resolve().parent
    candidates = [script_dir / ".env", script_dir.parent.parent / ".env"]
    env_path = next((p for p in candidates if p.exists()), None)
    if env_path is None:
        return

    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
        return
    except ImportError:
        pass

    # Minimal fallback parser.
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()


# LLM configuration — supports Anthropic and Anthropic-compatible endpoints
# (e.g. z.ai's GLM via https://api.z.ai/api/anthropic).
#
# Read from environment (optionally populated from a .env file, see above).
# Env vars:
#   LLM_API_TOKEN  — API key (falls back to ANTHROPIC_API_KEY)
#   LLM_BASE_URL   — custom Anthropic-compatible base URL (optional)
#   LLM_MODEL      — model id (default: claude-sonnet-4-20250514;
#                    for z.ai use e.g. glm-4.6)
#   LLM_TIMEOUT_MS — request timeout in milliseconds (default: 3000000)
DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_TIMEOUT_MS = 3000000

MODEL = os.environ.get("LLM_MODEL", DEFAULT_MODEL)
MAX_CONCURRENT = 5
QUERIES_PER_CHUNK = 3
MAX_RETRIES = 3
SYSTEM_PROMPT = "You are helping evaluate an emergency survival assistant."


def _parse_queries(raw: str) -> list[str] | None:
    """Extract and validate the queries list from a model response.

    Tolerant of markdown code fences, surrounding prose, and minor JSON noise:
    pulls the first balanced {...} object out of the text before parsing.
    Returns a cleaned list of query strings, or None if nothing usable.
    """
    if not raw:
        return None

    text = raw.strip()

    # Strip a leading ```json / ``` fence and trailing ``` if present.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Fall back to grabbing the first {...} block from the text.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        else:
            return None

    if not isinstance(data, dict):
        return None

    queries = data.get("queries", [])
    if not isinstance(queries, list):
        return None

    cleaned = [q.strip() for q in queries if isinstance(q, str) and len(q.strip()) > 5]
    return cleaned or None


def _build_prompt(pack_name: str, section_path: str, chunk_text: str) -> str:
    return f"""Given this knowledge chunk, generate 2-3 realistic questions a person in an emergency might type or speak.

Requirements:
- Urgent, colloquial phrasing (e.g. 'how do I stop bleeding')
- Each question answerable by this specific chunk
- Vary phrasing (direct questions, descriptive situations, short keywords)

Chunk from '{pack_name}' ({section_path}):
{chunk_text[:2000]}

Respond in JSON: {{"queries": ["q1", "q2", "q3"]}}"""


def load_packs(packs_dir: str) -> list[dict]:
    """Load all pack JSONs and return flat list of (chunk, pack_id, pack_name)."""
    packs_path = Path(packs_dir)
    entries = []

    for pf in sorted(packs_path.glob("*.json")):
        if pf.name == "packs_registry.json":
            continue
        with open(pf, "r", encoding="utf-8") as f:
            pack = json.load(f)

        pack_id = pack["packId"]
        pack_name = pack["packName"]

        for i, chunk in enumerate(pack["chunks"]):
            entries.append({
                "pack_id": pack_id,
                "pack_name": pack_name,
                "chunk_id": chunk["chunkId"],
                "chunk_text": chunk["text"],
                "section_path": chunk.get("sectionPath", ""),
                "chunk_idx": i,
            })

    return entries


async def generate_queries_for_chunk(
    client: anthropic.AsyncAnthropic,
    entry: dict,
    semaphore: asyncio.Semaphore,
) -> list[str] | None:
    """Generate queries for a single chunk via Claude API."""
    async with semaphore:
        prompt = _build_prompt(
            entry["pack_name"],
            entry["section_path"],
            entry["chunk_text"],
        )

        last_raw = ""
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=512,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                # Concatenate all text blocks (some providers split output).
                last_raw = "".join(
                    getattr(b, "text", "") for b in response.content
                ).strip()

                queries = _parse_queries(last_raw)
                if queries:
                    return queries
                # Empty / unparseable — fall through to retry.
            except Exception as e:
                last_raw = f"<API error: {e}>"

            if attempt < MAX_RETRIES - 1:
                # Brief backoff before retrying.
                await asyncio.sleep(1.0 * (attempt + 1))

        tqdm.write(
            f"  Failed for {entry['chunk_id']} after {MAX_RETRIES} attempts. "
            f"Last response: {last_raw[:200]!r}",
            file=sys.stderr,
        )
        return None


def load_existing_dataset(output_path: str) -> dict | None:
    """Load existing dataset for resume support."""
    if not os.path.exists(output_path):
        return None
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, KeyError):
        return None


async def main_async(args):
    api_key = os.environ.get("LLM_API_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "Error: set LLM_API_TOKEN (or ANTHROPIC_API_KEY) environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    base_url = os.environ.get("LLM_BASE_URL")  # None → Anthropic default
    timeout_ms = float(os.environ.get("LLM_TIMEOUT_MS", DEFAULT_TIMEOUT_MS))
    timeout_s = timeout_ms / 1000.0

    client_kwargs = {"api_key": api_key, "timeout": timeout_s}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = anthropic.AsyncAnthropic(**client_kwargs)
    print(f"Model:    {MODEL}")
    print(f"Base URL: {base_url or 'https://api.anthropic.com (default)'}")
    print(f"Timeout:  {timeout_s:.0f}s")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    entries = load_packs(args.packs)
    print(f"Found {len(entries)} chunks across all packs.")

    # Resume support: load existing results
    existing = load_existing_dataset(args.output)
    done_chunk_ids: set[str] = set()
    queries: list[dict] = []

    if existing and "queries" in existing:
        for q in existing["queries"]:
            queries.append(q)
            for cid in q["relevant_chunk_ids"]:
                done_chunk_ids.add(cid)
        print(f"Resuming: {len(done_chunk_ids)} chunks already processed ({len(queries)} queries).")

    # Filter to unprocessed entries
    todo = [e for e in entries if e["chunk_id"] not in done_chunk_ids]
    print(f"Processing {len(todo)} remaining chunks...")

    # Process in batches
    batch_size = 50
    with tqdm(total=len(todo), unit="chunk", desc="Generating") as pbar:
        for batch_start in range(0, len(todo), batch_size):
            batch = todo[batch_start:batch_start + batch_size]

            tasks = [
                generate_queries_for_chunk(client, entry, semaphore)
                for entry in batch
            ]
            results = await asyncio.gather(*tasks)

            for entry, generated in zip(batch, results):
                if generated is None:
                    continue

                for qi, query_text in enumerate(generated[:QUERIES_PER_CHUNK]):
                    query_id = f"{entry['pack_id']}-{entry['chunk_idx']}-q{qi}"
                    queries.append({
                        "query_id": query_id,
                        "query": query_text,
                        "relevant_chunk_ids": [entry["chunk_id"]],
                        "pack_id": entry["pack_id"],
                    })

            pbar.update(len(batch))
            pbar.set_postfix(queries=len(queries))

            # Save intermediate results after each batch
            _save_dataset(args.output, queries)

    # Save final dataset
    _save_dataset(args.output, queries)
    print(f"\nDone. {len(queries)} queries written to {args.output}")


def _save_dataset(output_path: str, queries: list[dict]):
    """Save dataset to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset = {
        "version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator_model": MODEL,
        "total_queries": len(queries),
        "queries": queries,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic benchmark queries")
    parser.add_argument(
        "--packs", default="assets/rag/packs",
        help="Directory containing pack JSONs"
    )
    parser.add_argument(
        "--output", default="benchmark/dataset.json",
        help="Output path for benchmark dataset"
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent.parent
    args.packs = str((project_root / args.packs).resolve())
    args.output = str((project_root / args.output).resolve())

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
