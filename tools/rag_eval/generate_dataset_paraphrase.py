#!/usr/bin/env python3
"""Generate paraphrase-controlled queries to minimize n-gram overlap.

Takes existing sighted queries and rewrites them to minimize lexical overlap
with the source chunk while preserving intent. This produces an intermediate
leakage level between sighted and blind generation.

Usage:
    python tools/rag_eval/generate_dataset_paraphrase.py \
        --input benchmark/dataset.json \
        --packs assets/rag/packs \
        --output benchmark/dataset_paraphrase.json
"""
import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("Error: 'anthropic' package required.", file=sys.stderr)
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("Error: 'tqdm' package required.", file=sys.stderr)
    sys.exit(1)

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_dotenv():
    """Load .env file."""
    script_dir = Path(__file__).resolve().parent
    candidates = [script_dir / ".env", script_dir.parent.parent / ".env"]
    env_path = next((p for p in candidates if p.exists()), None)
    if env_path is None:
        return

    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
    except ImportError:
        pass


_load_dotenv()

MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514")
MAX_CONCURRENT = 5
MAX_RETRIES = 3
SYSTEM_PROMPT = "You are helping evaluate an emergency survival assistant."


def _build_rewrite_prompt(query: str, pack_name: str, section_path: str) -> str:
    return f"""Rewrite this emergency question to use different words and phrasing while keeping the same meaning.

Original question: "{query}"

Context: This question is about {pack_name} ({section_path}).

Requirements:
- Change vocabulary and sentence structure
- Keep the same emergency intent and information need
- Make it sound like something a different person might ask
- DO NOT make it more formal or more technical - keep it colloquial
- Respond in JSON: {{"rewritten_query": "your rewritten question"}}"""


def _parse_rewrite_response(raw: str) -> str | None:
    """Extract rewritten query from model response."""
    if not raw:
        return None

    text = raw.strip()

    # Strip code fences
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        else:
            return None

    rewritten = data.get("rewritten_query")
    if isinstance(rewritten, str) and len(rewritten.strip()) > 5:
        return rewritten.strip()
    return None


def load_packs(packs_dir: str) -> dict:
    """Load packs and return a chunk_id -> chunk mapping."""
    packs_path = Path(packs_dir)
    chunks = {}

    for pf in sorted(packs_path.glob("*.json")):
        if pf.name == "packs_registry.json":
            continue
        with open(pf, "r", encoding="utf-8") as f:
            pack = json.load(f)

        for chunk in pack["chunks"]:
            chunks[chunk["chunkId"]] = {
                "text": chunk["text"],
                "pack_name": pack["packName"],
                "section_path": chunk.get("sectionPath", ""),
            }

    return chunks


async def rewrite_query(
    client: anthropic.AsyncAnthropic,
    query: str,
    pack_name: str,
    section_path: str,
    semaphore: asyncio.Semaphore,
) -> str | None:
    """Rewrite a single query via Claude API."""
    async with semaphore:
        prompt = _build_rewrite_prompt(query, pack_name, section_path)

        for attempt in range(MAX_RETRIES):
            try:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=256,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )

                raw = "".join(getattr(b, "text", "") for b in response.content).strip()
                rewritten = _parse_rewrite_response(raw)

                if rewritten:
                    return rewritten

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))

        return None


async def main_async(args):
    api_key = os.environ.get("LLM_API_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: set LLM_API_TOKEN or ANTHROPIC_API_KEY", file=sys.stderr)
        sys.exit(1)

    base_url = os.environ.get("LLM_BASE_URL")
    timeout_s = float(os.environ.get("LLM_TIMEOUT_MS", "3000000")) / 1000.0

    client_kwargs = {"api_key": api_key, "timeout": timeout_s}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = anthropic.AsyncAnthropic(**client_kwargs)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Load input dataset
    print(f"Loading dataset: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    queries = dataset["queries"]
    print(f"Loaded {len(queries)} queries")

    # Load packs for context
    chunks = load_packs(args.packs)
    print(f"Loaded {len(chunks)} chunks from packs")

    # Check for resume
    existing = None
    if os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except:
            pass

    done_queries = set()
    rewritten_queries = []

    if existing and "queries" in existing:
        for q in existing["queries"]:
            rewritten_queries.append(q)
            done_queries.add(q["original_query_id"])
        print(f"Resuming: {len(done_queries)} queries already processed")

    todo = [q for q in queries if q["query_id"] not in done_queries]
    print(f"Processing {len(todo)} remaining queries...")

    # Process in batches
    batch_size = 50
    with tqdm(total=len(todo), unit="query", desc="Rewriting") as pbar:
        for batch_start in range(0, len(todo), batch_size):
            batch = todo[batch_start:batch_start + batch_size]

            tasks = []
            for q in batch:
                source_chunk_id = q["relevant_chunk_ids"][0]
                chunk_info = chunks.get(source_chunk_id, {})

                task = rewrite_query(
                    client, q["query"], chunk_info.get("pack_name", "Unknown"),
                    chunk_info.get("section_path", ""), semaphore
                )
                tasks.append((q, task))

            results = await asyncio.gather(*[t for _, t in tasks], return_exceptions=True)

            for (q, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    continue

                if result is None:
                    # Keep original if rewrite failed
                    rewritten_query = q["query"]
                else:
                    rewritten_query = result

                rewritten_queries.append({
                    "query_id": f"{q['query_id']}-paraphrase",
                    "original_query_id": q["query_id"],
                    "query": rewritten_query,
                    "relevant_chunk_ids": q["relevant_chunk_ids"],
                    "pack_id": q["pack_id"],
                })

            pbar.update(len(batch))

            # Save intermediate
            _save_output(args.output, rewritten_queries, dataset)

    _save_output(args.output, rewritten_queries, dataset)
    print(f"\nDone. {len(rewritten_queries)} rewritten queries saved to {args.output}")


def _save_output(output_path: str, queries: list, original_dataset: dict):
    """Save output dataset."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output = {
        "version": "1.1",
        "mode": "paraphrase",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator_model": MODEL,
        "source_dataset": original_dataset.get("generated_at", "unknown"),
        "source_model": original_dataset.get("generator_model", "unknown"),
        "total_queries": len(queries),
        "queries": queries,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Generate paraphrase-controlled queries")
    parser.add_argument("--input", default="benchmark/dataset.json", help="Input dataset")
    parser.add_argument("--packs", default="assets/rag/packs", help="Packs directory")
    parser.add_argument("--output", default="benchmark/dataset_paraphrase.json", help="Output path")

    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).resolve().parent.parent.parent
    args.input = str((project_root / args.input).resolve())
    args.packs = str((project_root / args.packs).resolve())
    args.output = str((project_root / args.output).resolve())

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
