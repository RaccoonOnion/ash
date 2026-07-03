#!/usr/bin/env python3
"""Estimate multi-positive expansion precision via an LLM judge.

Reads the full expansion audit set (every added gold chunk with its query +
text) and asks the configured Anthropic-compatible endpoint whether each ADDED
chunk is actually relevant to the query. Reports precision = relevant / total.

This is an LLM-estimated precision (not human) and is labelled as such. Reuses
the same .env / endpoint as the generators (LLM_API_TOKEN, LLM_BASE_URL,
LLM_MODEL).

Usage:
    python tools/rag_eval/audit_expansion_precision.py \\
        --audit benchmark/dataset_multi_positive_audit_full.json \\
        --output benchmark/expansion_precision_audit.json
"""
import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("Error: 'anthropic' package required.", file=sys.stderr); sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load_dotenv():
    script_dir = Path(__file__).resolve().parent
    for p in (script_dir / ".env", script_dir.parent.parent / ".env"):
        if p.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(p, override=False)
                return
            except ImportError:
                pass


_load_dotenv()

MODEL = os.environ.get("LLM_MODEL", "claude-sonnet-4-20250514")
MAX_CONCURRENT = 5
MAX_RETRIES = 3


def _judge_prompt(query: str, added_text: str) -> str:
    return f"""You are auditing a retrieval-gold-label expansion step. A system marked the chunk below as an ADDITIONAL relevant gold label for the question. Judge whether that is correct.

Question: {query}

Candidate chunk (a different chunk proposed as additional gold):
\"\"\"
{added_text[:800]}
\"\"\"

Does this candidate chunk contain information that is genuinely relevant to answering the question? Ignore mere topical adjacency — it must actually help answer or address the question's specific need.

Answer with ONLY one word: yes or no."""


def _parse_yesno(raw: str) -> bool | None:
    if not raw:
        return None
    t = raw.strip().lower()
    m = re.search(r"\b(yes|no|true|false)\b", t)
    if not m:
        return None
    return m.group(1) in {"yes", "true"}


async def judge_one(client, pair, sem):
    async with sem:
        prompt = _judge_prompt(pair["query"], pair["added_text"])
        last = ""
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.messages.create(
                    model=MODEL, max_tokens=8,
                    messages=[{"role": "user", "content": prompt}],
                )
                last = "".join(getattr(b, "text", "") for b in resp.content).strip()
                verdict = _parse_yesno(last)
                if verdict is not None:
                    return verdict
            except Exception as e:
                last = f"<err {e}>"
            await asyncio.sleep(0.5 * (attempt + 1))
        return None  # undecided after retries


async def main_async(args):
    api_key = os.environ.get("LLM_API_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: set LLM_API_TOKEN / ANTHROPIC_API_KEY", file=sys.stderr); sys.exit(1)
    base_url = os.environ.get("LLM_BASE_URL")
    kw = {"api_key": api_key, "timeout": 120.0}
    if base_url:
        kw["base_url"] = base_url
    client = anthropic.AsyncAnthropic(**kw)

    audit = json.load(open(args.audit))
    pairs = audit["pairs"]
    print(f"Model: {MODEL} | judging {len(pairs)} expansion pairs ...")

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    judged = []
    yes = no = undecided = 0
    # Stream progress in batches.
    batch = 50
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        res = await asyncio.gather(*(judge_one(client, p, sem) for p in chunk))
        for p, v in zip(chunk, res):
            if v is True:
                yes += 1
            elif v is False:
                no += 1
            else:
                undecided += 1
            judged.append({**{k: p[k] for k in ("query_id", "query", "added_chunk_id")}, "relevant": v})
        print(f"  {min(i + batch, len(pairs))}/{len(pairs)}  yes={yes} no={no} undecided={undecided}")

    decided = yes + no
    precision = yes / decided if decided else None
    summary = {
        "method": "LLM-judge (model=%s); relevance of each ADDED gold chunk vs its query" % MODEL,
        "n_pairs": len(pairs),
        "n_relevant_yes": yes,
        "n_irrelevant_no": no,
        "n_undecided": undecided,
        "precision": precision,
        "precision_note": "yes / (yes + no); undecided excluded from denominator",
    }
    out = {"summary": summary, "judgments": judged}
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("\n=== Precision ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nWritten to: {args.output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="benchmark/dataset_multi_positive_audit_full.json")
    ap.add_argument("--output", default="benchmark/expansion_precision_audit.json")
    args = ap.parse_args()
    project_root = Path(__file__).resolve().parent.parent.parent
    args.audit = str((project_root / args.audit).resolve())
    args.output = str((project_root / args.output).resolve())
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
