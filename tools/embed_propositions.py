#!/usr/bin/env python3
"""Embed LLM-extracted Q&A propositions and rewrite pack JSONs.

Reads `tools/propositions/{packId}.json` (produced by the proposition-
extraction agents) and converts each proposition into a chunk in the
existing pack-JSON format the Flutter app already knows how to load.

Key design decision: the question is stored as the FIRST `####` line of
the chunk text, with the answer following. This way the app's existing
`extractChunkHeading` path surfaces the question as the chunk's title in
the Library reader and as the citation chip label — no app-side schema
migration needed for ChunkEntity.

Usage:
    cd <repo root>
    source tools/.venv/bin/activate
    python tools/embed_propositions.py
"""
import json
import re
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ─── Answer polish ───────────────────────────────────────────────────────────
#
# The HazAdapt source markdown has a handful of recurring quirks that render
# awkwardly in the app — single-item numbered lists ("1." with no "2."),
# blockquote-wrapped headings, dead internal `hazadapt://` links, list items
# separated by blank lines so the renderer restarts numbering at 1, etc.
# These are visual / formatting issues; the content is correct. We clean them
# up at build time so the on-device card stays readable.

def _strip_hazadapt_links(s: str) -> str:
    """`[text](hazadapt://...)` → `text`. The custom URI scheme has no
    handler in the app today; rendering a blue underlined link that does
    nothing on tap is worse than just showing the text inline."""
    return re.sub(r"\[([^\]]*)\]\(hazadapt://[^\)]*\)", r"\1", s)


def _normalize_bold_italic(s: str) -> str:
    """Collapse mismatched `***` markers down to `**`. HazAdapt's authors
    frequently write `***EVALUATE:** Quickly decide...` — three stars open,
    two stars close — which renders as italic-bold with a dangling italic.
    Demoting to plain bold loses the italic flourish but keeps the
    important emphasis and avoids the rendering glitch.

    Then strip an orphan trailing `*` left by patterns like
    `**Label:** body content *` (the residue of `***Label:** body *`
    after the ***→** collapse). Conservative: only strips when the line
    has a `**X:**` lead-in and a single dangling `*` at the end."""
    s = re.sub(r"\*{3,}", "**", s)
    s = re.sub(
        r"(\*\*[^\*\n]+?:\*\*\s*[^\*\n]+?)\s*\*(?=\s*(?:\n|$))",
        r"\1",
        s,
        flags=re.MULTILINE,
    )
    return s


def _unwrap_whole_answer_blockquote(s: str) -> str:
    """If literally every non-blank line of the answer starts with `>`,
    the blockquote is just a formatting habit, not a semantic callout.
    Strip the leading `>` from every line."""
    lines = s.split("\n")
    nonblank = [l for l in lines if l.strip()]
    if not nonblank:
        return s
    if all(l.lstrip().startswith(">") for l in nonblank):
        return "\n".join(re.sub(r"^>\s?", "", l) for l in lines)
    return s


def _flatten_blockquote_headings(s: str) -> str:
    """`> #### Heading` → `#### Heading`. Headings inside blockquotes look
    like indented section titles rather than the proper level — the leading
    `>` is almost always an accidental nesting in the source."""
    return re.sub(r"^>\s*(#{1,6}\s)", r"\1", s, flags=re.MULTILINE)


def _demote_lone_numbered(s: str) -> str:
    """If the answer contains exactly one numbered-list item, strip the
    `N. ` prefix. A lone "1." with no "2." misleads the user into thinking
    the answer is incomplete (the screenshot-reported case)."""
    list_starts = re.findall(r"^\s*\d+\.\s", s, flags=re.MULTILINE)
    if len(list_starts) == 1:
        return re.sub(r"^(\s*)\d+\.\s+", r"\1", s, count=1, flags=re.MULTILINE)
    return s


def _tighten_numbered_lists(s: str) -> str:
    """Collapse blank lines between consecutive numbered items so the
    renderer treats them as one list. HazAdapt frequently writes
    `1. Foo\n\n2. Bar\n\n3. Baz` — the blank lines break the list and the
    renderer restarts numbering. Removing them keeps `1, 2, 3` contiguous."""
    lines = s.split("\n")
    num_pat = re.compile(r"^\s*\d+\.\s")
    out: list[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        if (
            cur.strip() == ""
            and 0 < i < len(lines) - 1
            and num_pat.match(lines[i - 1])
            and num_pat.match(lines[i + 1])
        ):
            # Drop this blank line — it's splitting one numbered list.
            i += 1
            continue
        out.append(cur)
        i += 1
    return "\n".join(out)


def _drop_empty_headings(s: str) -> str:
    """Drop a heading line if its body is empty (the next non-blank line
    is another heading or end-of-answer). The source sometimes has stray
    `#### Section` lines that got de-bodied during chunking."""
    lines = s.split("\n")
    out: list[str] = []
    heading_pat = re.compile(r"^#{1,6}\s+\S")
    for i, line in enumerate(lines):
        if heading_pat.match(line):
            # Look ahead for body content before the next heading.
            has_body = False
            for nxt in lines[i + 1:]:
                if not nxt.strip():
                    continue
                if heading_pat.match(nxt):
                    break
                has_body = True
                break
            if not has_body:
                continue  # drop this empty heading
        out.append(line)
    return "\n".join(out)


def _strip_leading_separators(s: str) -> str:
    """Drop leading horizontal rules (`---` / `***`) and blank lines.
    These survive earlier sanitization but only ever appear at chunk
    boundaries as separators inherited from the source."""
    lines = s.split("\n")
    while lines and (
        not lines[0].strip()
        or re.match(r"^(?:-{3,}|\*{3,})\s*$", lines[0].strip())
    ):
        lines.pop(0)
    return "\n".join(lines)


def _collapse_blank_runs(s: str) -> str:
    """Collapse runs of 3+ newlines to a single paragraph break."""
    return re.sub(r"\n{3,}", "\n\n", s)


def polish_answer(text: str) -> str:
    """Run every cleanup pass over a proposition's answer body. Order
    matters: link strip and bold-italic normalization touch character-level
    content; blockquote handling sits above; numbered-list cleanup sits
    above empty-heading and blank-collapse cleanup (which depend on the
    final structural shape)."""
    s = text
    s = _strip_hazadapt_links(s)
    s = _normalize_bold_italic(s)
    s = _unwrap_whole_answer_blockquote(s)
    s = _flatten_blockquote_headings(s)
    s = _demote_lone_numbered(s)
    s = _tighten_numbered_lists(s)
    s = _drop_empty_headings(s)
    s = _strip_leading_separators(s)
    s = _collapse_blank_runs(s)
    return s.strip()

REPO_ROOT = Path(__file__).parent.parent
PACKS_DIR = REPO_ROOT / "assets" / "rag" / "packs"
PROPS_DIR = REPO_ROOT / "tools" / "propositions"
REGISTRY = PACKS_DIR / "packs_registry.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_VERSION = "all-MiniLM-L6-v2-r5"


def _slug(s: str) -> str:
    """URL/filename-safe slug from a free-form string. Capped so chunkIds
    don't become absurdly long when a question is a full sentence."""
    s = re.sub(r"[^a-z0-9-]+", "-", s.lower()).strip("-")
    return s[:48] or "q"


def main() -> int:
    if not PROPS_DIR.exists():
        print(f"Proposition dir not found: {PROPS_DIR}", file=sys.stderr)
        return 1

    prop_files = sorted(PROPS_DIR.glob("*.json"))
    if not prop_files:
        print(f"No proposition files in {PROPS_DIR}", file=sys.stderr)
        return 1
    print(f"Found {len(prop_files)} proposition files")

    print(f"Loading {MODEL_NAME}…")
    model = SentenceTransformer(MODEL_NAME)
    print(f"  embedding dim: {model.get_embedding_dimension()}")

    size_by_id: dict[str, tuple[int, int]] = {}
    total_props = 0

    for pf in tqdm(prop_files, desc="Embedding+writing", unit="pack"):
        with pf.open() as f:
            data = json.load(f)
        pack_id = data["packId"]
        pack_name = data["packName"]
        propositions = data.get("propositions", [])
        if not propositions:
            continue
        total_props += len(propositions)

        # Embed all questions in one batch. normalize_embeddings=True
        # matches the runtime path (mean pool + L2 normalize) so the
        # vectors live in the same space as the on-device query embeddings.
        questions = [p["question"] for p in propositions]
        embeddings = model.encode(
            questions,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        chunks = []
        for i, (prop, emb) in enumerate(zip(propositions, embeddings)):
            q = prop["question"].strip()
            a = polish_answer(prop["answer"])
            # `#### {question}\n\n{answer}` lets the runtime
            # `extractChunkHeading` lift the question as the card title /
            # citation chip label without any schema change.
            text = f"#### {q}\n\n{a}"
            chunks.append({
                "chunkId": f"{pack_id}-{_slug(q)}-{i}",
                "text": text,
                "source": pack_id,
                "sectionPath": prop.get("sectionPath", ""),
                "sourceType": "core",
                "embeddingModelVersion": EMBEDDING_MODEL_VERSION,
                "embedding": [round(float(x), 6) for x in emb.tolist()],
            })

        out = {
            "packId": pack_id,
            "packName": pack_name,
            "version": "1.1",
            "embeddingModel": "all-MiniLM-L6-v2",
            "embeddingModelVersion": EMBEDDING_MODEL_VERSION,
            "sourceType": "core",
            "chunks": chunks,
        }

        out_path = PACKS_DIR / f"{pack_id}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
        size_by_id[pack_id] = (len(chunks), out_path.stat().st_size)

    # Update registry with new counts + sizes so the Flutter app shows
    # truthful entry counts in the Library / Store cards.
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

    print(f"\nDone. {total_props} propositions across {len(size_by_id)} packs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
