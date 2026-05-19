#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert train-only supervised mining output into master-lexicon-compatible rows.

Input is usually:
  outputs/lexicon/train_supervised_mining/train_supervised_ngram_features.csv

Output can be reviewed and then merged into the frame lexicon.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


TARGET_MAP = {"UP": "UP", "DOWN": "DOWN", "NEUTRAL": "NEUTRAL"}


def to_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def infer_frame(target: str, bucket: str, feature_type: str, cue: str) -> str:
    cue_l = cue.lower()
    if target == "NEUTRAL" or bucket == "neutralizer":
        if any(x in cue for x in ["教程", "攻略", "整理", "合集", "步骤", "怎么"]):
            return "tutorial_information"
        if any(x in cue for x in ["测评", "参数", "官网", "下载", "价格", "产品", "工具"]):
            return "product_tool_review"
        return "ordinary_information_or_daily_neutralizer"
    if target == "UP":
        if any(x in cue_l for x in ["offer", "录取", "上岸", "留学", "博士", "实习", "面试"]):
            return "achievement_elite_education"
        if any(x in cue for x in ["旅行", "演唱会", "海外", "港", "澳", "台", "solo", "圆梦"]):
            return "high_resource_lifestyle"
        if any(x in cue for x in ["变美", "瘦", "身材", "颜值", "穿搭", "被夸", "马甲线"]):
            return "appearance_body_success"
        if any(x in cue for x in ["隐藏款", "限量", "抢到", "谷子", "周边", "盲盒"]):
            return "scarce_fandom_consumption"
        return "implicit_upward_advantage"
    if target == "DOWN":
        if any(x in cue for x in ["父母", "妈妈", "爸爸", "家里", "原生家庭", "道德绑架"]):
            return "family_oppression_low_support"
        if any(x in cue_l for x in ["offer", "录取", "上岸", "被拒", "失败", "找不到", "没过"]):
            return "blocked_aspiration_failure"
        if any(x in cue for x in ["没办法", "只能", "被迫", "不敢", "无法", "不能"]):
            return "low_agency_constraint"
        if any(x in cue for x in ["焦虑", "自卑", "崩溃", "窒息", "委屈", "难受"]):
            return "negative_affect_social_pain"
        return "implicit_downward_disadvantage"
    return f"supervised_{feature_type}"


def infer_cue_type(target: str, bucket: str, feature_type: str) -> str:
    if bucket == "neutralizer" or target == "NEUTRAL":
        return "neutralizer_frame"
    if feature_type == "hash":
        return "domain_frame"
    if bucket == "ambiguous_or_review":
        return "relational_marker"
    return "domain_frame"


def infer_weight(bucket: str, purity: float, z: float) -> int:
    if bucket in {"core_cue", "neutralizer"} and purity >= 0.72 and z >= 2.0:
        return 3
    if purity >= 0.58 and z >= 1.5:
        return 2
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert supervised train mining features to lexicon candidate rows.")
    parser.add_argument("--features", type=Path, default=Path("outputs/lexicon/train_supervised_mining/train_supervised_ngram_features.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/lexicon/train_supervised_mining/train_supervised_lexicon_candidates.csv"))
    parser.add_argument("--min_purity", type=float, default=0.58)
    parser.add_argument("--min_coverage", type=int, default=5)
    parser.add_argument("--min_abs_log_odds_z", type=float, default=1.5)
    parser.add_argument("--include_ambiguous", action="store_true")
    parser.add_argument("--max_rows_per_label", type=int, default=800)
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    rows: List[Dict[str, Any]] = []
    kept_per_label = {"UP": 0, "DOWN": 0, "NEUTRAL": 0, "AMBIGUOUS": 0}

    for row in df.itertuples(index=False):
        target = str(getattr(row, "suggested_target", "")).strip().upper()
        if target not in TARGET_MAP:
            continue
        cue = str(getattr(row, "term", "")).strip()
        bucket = str(getattr(row, "suggested_bucket", "")).strip()
        feature_type = str(getattr(row, "feature_type", "")).strip()
        purity = to_float(getattr(row, "purity", 0.0))
        coverage = to_int(getattr(row, "coverage", 0))
        z = abs(to_float(getattr(row, "max_abs_log_odds_z", 0.0)))
        if not cue or purity < args.min_purity or coverage < args.min_coverage or z < args.min_abs_log_odds_z:
            continue
        out_target = target
        if bucket == "ambiguous_or_review":
            if not args.include_ambiguous:
                continue
            out_target = "AMBIGUOUS"
        if kept_per_label.get(out_target, 0) >= args.max_rows_per_label:
            continue

        count_up = to_int(getattr(row, "count_UP", 0))
        count_neu = to_int(getattr(row, "count_NEUTRAL", 0))
        count_down = to_int(getattr(row, "count_DOWN", 0))
        counts = {"UP": count_up, "DOWN": count_down, "NEUTRAL": count_neu}
        empirical_label = max(counts, key=counts.get)
        total = sum(counts.values())
        empirical_share = round(counts[empirical_label] / total, 3) if total else ""

        rows.append({
            "target_label": out_target,
            "frame": infer_frame(target, bucket, feature_type, cue),
            "cue": cue,
            "cue_type": infer_cue_type(target, bucket, feature_type),
            "source_basis": "train_supervised_logodds_chi2_mi_purity",
            "rationale": f"Train-only supervised feature; bucket={bucket}; purity={purity}; coverage={coverage}; max_abs_log_odds_z={z}.",
            "weight_1_3": infer_weight(bucket, purity, z),
            "context_rule": str(getattr(row, "context_rule_hint", "")).strip(),
            "count_UP": count_up,
            "count_DOWN": count_down,
            "count_NEUTRAL": count_neu,
            "empirical_label": empirical_label,
            "empirical_share": empirical_share,
            "supervised_bucket": bucket,
            "max_abs_log_odds_z": z,
            "max_chi2": getattr(row, "max_chi2", ""),
            "max_mi": getattr(row, "max_mi", ""),
        })
        kept_per_label[out_target] = kept_per_label.get(out_target, 0) + 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target_label", "frame", "cue", "cue_type", "source_basis", "rationale",
        "weight_1_3", "context_rule", "count_UP", "count_DOWN", "count_NEUTRAL",
        "empirical_label", "empirical_share", "supervised_bucket", "max_abs_log_odds_z",
        "max_chi2", "max_mi",
    ]
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print({
        "rows": len(rows),
        "output": str(args.output),
        "kept_per_label": kept_per_label,
    })


if __name__ == "__main__":
    main()
