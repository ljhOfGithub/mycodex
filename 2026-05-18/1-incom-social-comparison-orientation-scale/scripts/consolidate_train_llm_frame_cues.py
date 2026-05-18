#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Consolidate JSONL from llm_extract_train_frame_cues.py into reviewable lexicon
candidates and optional master-lexicon-compatible rows.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


LABEL_TO_TARGET = {"UPWARD": "UP", "DOWNWARD": "DOWN", "NEUTRAL": "NEUTRAL"}
TARGET_TO_COUNT_COL = {"UP": "count_UP", "DOWN": "count_DOWN", "NEUTRAL": "count_NEUTRAL", "AMBIGUOUS": "count_NEUTRAL"}


def norm_target(x: Any) -> str:
    s = str(x or "").strip().upper()
    if s in ["UPWARD", "UP"]:
        return "UP"
    if s in ["DOWNWARD", "DOWN"]:
        return "DOWN"
    if s in ["NEUTRAL", "NEU"]:
        return "NEUTRAL"
    if s == "AMBIGUOUS":
        return "AMBIGUOUS"
    return ""


def norm_weight(x: Any) -> int:
    try:
        value = int(float(x))
    except Exception:
        value = 2
    return max(1, min(3, value))


def iter_records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def collect_candidates(jsonl_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rec in iter_records(jsonl_path):
        parsed = rec.get("parsed", {})
        if not isinstance(parsed, dict):
            continue
        post_id = str(rec.get("id", ""))
        gold_label = str(rec.get("gold_label", ""))
        content = str(rec.get("content", ""))

        for cand in parsed.get("candidate_cues", []) or []:
            if not isinstance(cand, dict):
                continue
            cue = str(cand.get("cue", "")).strip()
            frame = str(cand.get("frame", "")).strip()
            target_raw = str(cand.get("target_label", "")).strip()
            target = norm_target(target_raw)
            target_fallback_reason = ""
            if not target:
                target = LABEL_TO_TARGET.get(gold_label, "")
                target_fallback_reason = f"invalid_target_label:{target_raw}"
            if not cue or not frame or not target:
                continue
            rows.append({
                "target_label": target,
                "target_label_original": target_raw,
                "target_label_fallback_reason": target_fallback_reason,
                "frame": frame,
                "cue": cue,
                "cue_type": str(cand.get("cue_type", "domain_frame")).strip() or "domain_frame",
                "rationale": str(cand.get("rationale", "")).strip(),
                "context_rule": str(cand.get("context_rule", "")).strip(),
                "weight_1_3": norm_weight(cand.get("weight_1_3", 2)),
                "confidence": cand.get("confidence", ""),
                "source_post_id": post_id,
                "source_gold_label": gold_label,
                "source_content": content,
            })
    return rows


def aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["target_label"], row["frame"], row["cue"])
        grouped[key].append(row)

    out: List[Dict[str, Any]] = []
    for (target, frame, cue), items in grouped.items():
        cue_type = Counter([x["cue_type"] for x in items if x.get("cue_type")]).most_common(1)
        rationale = next((x["rationale"] for x in items if x.get("rationale")), "")
        context_rule = next((x["context_rule"] for x in items if x.get("context_rule")), "")
        weights = [norm_weight(x.get("weight_1_3", 2)) for x in items]
        confs = pd.to_numeric(pd.Series([x.get("confidence", "") for x in items]), errors="coerce").dropna()
        gold_counts = Counter([x.get("source_gold_label", "") for x in items])

        count_up = int(gold_counts.get("UPWARD", 0))
        count_down = int(gold_counts.get("DOWNWARD", 0))
        count_neutral = int(gold_counts.get("NEUTRAL", 0))
        total = count_up + count_down + count_neutral
        count_map = {"UP": count_up, "DOWN": count_down, "NEUTRAL": count_neutral}
        empirical_label = max(count_map, key=count_map.get) if total else ""
        empirical_share = round(count_map[empirical_label] / total, 3) if total else ""

        out.append({
            "target_label": target,
            "frame": frame,
            "cue": cue,
            "cue_type": cue_type[0][0] if cue_type else "domain_frame",
            "source_basis": "LLM_train_extraction+XHS",
            "rationale": rationale,
            "weight_1_3": max(weights) if weights else 2,
            "context_rule": context_rule,
            "count_UP": count_up,
            "count_DOWN": count_down,
            "count_NEUTRAL": count_neutral,
            "empirical_label": empirical_label,
            "empirical_share": empirical_share,
            "llm_docfreq": len(items),
            "mean_confidence": round(float(confs.mean()), 3) if len(confs) else "",
            "example_post_ids": "|".join([x["source_post_id"] for x in items[:5]]),
        })

    out.sort(key=lambda r: (-int(r["llm_docfreq"]), r["target_label"], r["frame"], r["cue"]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate LLM train frame/cue extractions.")
    parser.add_argument("--input_jsonl", type=Path, default=Path("outputs/lexicon/train_llm_extractions/train_frame_cues.jsonl"))
    parser.add_argument("--raw_output", type=Path, default=Path("outputs/lexicon/train_llm_extractions/train_frame_cues_raw.csv"))
    parser.add_argument("--candidate_output", type=Path, default=Path("outputs/lexicon/train_llm_extractions/train_frame_cue_candidates.csv"))
    parser.add_argument("--min_docfreq", type=int, default=2)
    parser.add_argument("--min_confidence", type=float, default=0.0)
    args = parser.parse_args()

    rows = collect_candidates(args.input_jsonl)
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)

    raw_fields = ["target_label", "target_label_original", "target_label_fallback_reason", "frame", "cue", "cue_type", "rationale", "context_rule", "weight_1_3", "confidence", "source_post_id", "source_gold_label", "source_content"]
    with args.raw_output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=raw_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    candidates = aggregate(rows)
    filtered: List[Dict[str, Any]] = []
    for row in candidates:
        if int(row["llm_docfreq"]) < args.min_docfreq:
            continue
        conf = row.get("mean_confidence", "")
        if conf != "" and float(conf) < args.min_confidence:
            continue
        filtered.append(row)

    cand_fields = [
        "target_label", "frame", "cue", "cue_type", "source_basis", "rationale",
        "weight_1_3", "context_rule", "count_UP", "count_DOWN", "count_NEUTRAL",
        "empirical_label", "empirical_share", "llm_docfreq", "mean_confidence",
        "example_post_ids",
    ]
    with args.candidate_output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cand_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(filtered)

    print(json.dumps({
        "raw_rows": len(rows),
        "candidate_rows": len(candidates),
        "filtered_rows": len(filtered),
        "raw_output": str(args.raw_output),
        "candidate_output": str(args.candidate_output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
