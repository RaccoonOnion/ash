#!/usr/bin/env python3
"""Analyze P4 retrieval-eval results: metrics table, leakage deltas, Kendall's tau.

Reads every benchmark/results_p4/<condition>_<mode>.json produced by run_eval.py
and reports:

1. Main metrics table — for each metric, a condition (sighted/blind/paraphrase/
   multipos) x mode (bm25/vector/hybrid) matrix of overall scores.
2. Leakage deltas — sighted minus blind / paraphrase, per metric and mode, i.e.
   how much each leakage-control condition depresses the score relative to the
   (leaky) sighted baseline.
3. Kendall's tau-b — per-pack rank correlation across conditions, per mode. High
   tau means the relative difficulty ranking of packs is stable when you swap
   leakage condition (methodology is robust); low tau means leakage distorts it.

Kendall's tau is implemented in pure Python (no scipy dependency).

Usage:
    python tools/rag_eval/analyze_results.py \\
        --results-dir benchmark/results_p4 \\
        --output benchmark/results_p4/summary.json
"""
import argparse
import json
import math
from pathlib import Path

# Metrics reported in run_eval overall (order chosen for readable tables).
METRIC_KEYS = [
    "mrr",
    "ndcg@5",
    "ndcg@10",
    "hit_rate@1",
    "hit_rate@3",
    "hit_rate@5",
    "ragas_id_recall",
    "ragas_id_precision",
]
# Baseline (leaky) condition against which deltas / tau are measured.
BASELINE = "sighted"


def kendall_tau_b(x: list[float], y: list[float]) -> float | None:
    """Kendall's rank correlation coefficient (tau-b), tie-corrected.

    Returns None if there are fewer than 2 paired values or it is undefined
    (zero denominator).
    """
    n = len(x)
    if n < 2 or len(y) != n:
        return None

    concordant = discordant = 0
    tx = ty = 0  # tied pairs (only-x, only-y)

    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            sx = 1 if dx > 0 else (-1 if dx < 0 else 0)
            sy = 1 if dy > 0 else (-1 if dy < 0 else 0)
            if sx == 0 or sy == 0:
                if sx == 0 and sy != 0:
                    tx += 1
                if sy == 0 and sx != 0:
                    ty += 1
                continue  # tied on at least one side: not concordant/discordant
            if sx == sy:
                concordant += 1
            else:
                discordant += 1

    denom = math.sqrt((concordant + discordant + tx) * (concordant + discordant + ty))
    if denom == 0:
        return None
    return (concordant - discordant) / denom


def load_results(results_dir: Path) -> dict:
    """Return {condition: {mode: result_dict}} parsed from filenames."""
    out: dict[str, dict[str, dict]] = {}
    for p in sorted(results_dir.glob("*.json")):
        if p.name in {"summary.json"}:
            continue
        stem = p.stem
        if "_" not in stem:
            continue
        condition, mode = stem.rsplit("_", 1)
        if mode not in {"bm25", "vector", "hybrid"}:
            continue
        with open(p, "r", encoding="utf-8") as f:
            out.setdefault(condition, {})[mode] = json.load(f)
    return out


def fmt(v, w=7, prec=4):
    if v is None:
        return f"{'n/a':>{w}}"
    if isinstance(v, float):
        return f"{v:>{w}.{prec}f}"
    return f"{str(v):>{w}}"


def print_metric_tables(data: dict, modes: list[str], conditions: list[str]):
    print("=" * 72)
    print("  MAIN METRICS  (overall; rows=condition, cols=retrieval mode)")
    print("=" * 72)
    for mk in METRIC_KEYS:
        print(f"\n  {mk}:")
        header = "  " + fmt("condition", 12) + "".join(fmt(m, 9) for m in modes)
        print(header)
        for cond in conditions:
            row = data.get(cond, {})
            cells = []
            for m in modes:
                o = row.get(m, {}).get("overall", {})
                cells.append(fmt(o.get(mk), 9))
            print("  " + fmt(cond, 12) + "".join(cells))


def print_deltas(data: dict, modes: list[str], conditions: list[str]):
    base = data.get(BASELINE, {})
    others = [c for c in conditions if c != BASELINE and c != "multipos"]
    if not base or not others:
        print("\n  (deltas need a sighted baseline + >=1 leakage condition; skipping)")
        return
    print("\n" + "=" * 72)
    print(f"  LEAKAGE DELTAS  ({BASELINE} - condition); positive = score drops under control")
    print("=" * 72)
    for mk in METRIC_KEYS:
        print(f"\n  {mk}:")
        print("  " + fmt("control", 12) + "".join(fmt(m, 9) for m in modes))
        for cond in others:
            cells = []
            for m in modes:
                b = base.get(m, {}).get("overall", {}).get(mk)
                c = data.get(cond, {}).get(m, {}).get("overall", {}).get(mk)
                d = (b - c) if (b is not None and c is not None) else None
                cells.append(fmt(d, 9))
            print("  " + fmt(cond, 12) + "".join(cells))


def print_kendall(data: dict, modes: list[str], conditions: list[str]):
    base = data.get(BASELINE, {})
    others = [c for c in conditions if c != BASELINE and c != "multipos"]
    if not base or not others:
        print("\n  (Kendall's tau needs sighted + >=1 leakage condition; skipping)")
        return
    print("\n" + "=" * 72)
    print(f"  KENDALL'S tau-b  (per-pack rank correlation vs {BASELINE}, by mode)")
    print("  High tau => pack-difficulty ranking stable across leakage conditions.")
    print("=" * 72)
    print("  " + fmt("control", 12) + fmt("metric", 16) + "".join(fmt(m, 9) for m in modes))
    for cond in others:
        for mk in ["mrr", "ndcg@10", "hit_rate@5"]:
            cells = []
            for m in modes:
                bp = base.get(m, {}).get("per_pack", {})
                cp = data.get(cond, {}).get(m, {}).get("per_pack", {})
                common = sorted(set(bp) & set(cp))
                if len(common) < 3:
                    cells.append(fmt(None, 9))
                    continue
                xs = [bp[pk].get(mk, 0.0) for pk in common]
                ys = [cp[pk].get(mk, 0.0) for pk in common]
                cells.append(fmt(kendall_tau_b(xs, ys), 9))
            print("  " + fmt(cond, 12) + fmt(mk, 16) + "".join(cells))


def main():
    ap = argparse.ArgumentParser(description="Analyze P4 retrieval-eval results")
    ap.add_argument("--results-dir", default="benchmark/results_p4")
    ap.add_argument("--output", default="benchmark/results_p4/summary.json")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    results_dir = (project_root / args.results_dir).resolve()
    data = load_results(results_dir)

    if not data:
        print(f"No result files found in {results_dir}", flush=True)
        return

    # Stable, readable order; multipos last (different gold, not a leakage step).
    order = ["sighted", "blind", "paraphrase", "multipos"]
    conditions = [c for c in order if c in data] + sorted(set(data) - set(order))
    modes = [m for m in ["bm25", "vector", "hybrid"] if any(m in data[c] for c in conditions)]

    print(f"\nLoaded {sum(len(v) for v in data.values())} runs from {results_dir}")
    print(f"Conditions: {conditions} | Modes: {modes}\n")

    print_metric_tables(data, modes, conditions)
    print_deltas(data, modes, conditions)
    print_kendall(data, modes, conditions)

    # Compact machine-readable summary (overall metrics only).
    summary = {
        c: {m: {k: data[c][m]["overall"].get(k) for k in METRIC_KEYS}
            for m in modes if m in data.get(c, {})}
        for c in conditions
    }
    out_path = (project_root / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary written to: {out_path}")


if __name__ == "__main__":
    main()
