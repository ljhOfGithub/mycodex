#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train-only supervised lexical mining for XHS social-comparison cues.

This script complements the LLM train extraction pipeline with mature
supervised lexical statistics:

- log odds ratio with informative Dirichlet prior
- one-vs-rest chi-square
- one-vs-rest mutual information
- contrastive cue purity / coverage / ambiguity
- optional L1 logistic regression coefficients when sklearn is installed
- train-internal hard-negative mining via cross-validated Bernoulli NB
- regex-based negation / blocked aspiration pattern mining
- seed expansion candidates by train-only lexical similarity

It never reads val/test and never calls an API.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import re
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


LABEL_NAMES = {0: "UP", 1: "NEUTRAL", 2: "DOWN"}
TARGET_TO_CODE = {"UP": 0, "UPWARD": 0, "NEUTRAL": 1, "NEU": 1, "DOWN": 2, "DOWNWARD": 2}

STOP_CHARS = set("的一是在不了和有就都也很还我你他她它们这那啊呀呢吗吧被把与及或啦呢嘛了哈")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
HASHTAG_RE = re.compile(r"#[^#\s]{1,40}")
LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+.-]{1,30}")
DROP_TERM_RE = re.compile(r"^[0-9\s\W_]+$")

NEGATION_PATTERNS: List[Tuple[str, str]] = [
    ("negated_achievement", r"(没|没有|未|无|别|不|没能|不能|无法).{0,8}(offer|oc|录取|上岸|过面|入职|实习|毕业|拿到|收到)"),
    ("blocked_aspiration", r"(想|打算|准备|希望|本来想).{0,12}(留学|上岸|考研|考公|面试|入职|变美|瘦|减肥|去).{0,12}(但|但是|结果|却|可惜|失败|没|被拒|不行|不能|无法)"),
    ("self_other_gap", r"(别人|同龄人|身边人|朋友|室友|同学).{0,12}(都|已经|全|也).{0,12}(我|自己).{0,12}(却|还|没|没有|只能|不行)"),
    ("appearance_anxiety", r"(容貌|身材|体重|脸|腿|腰|肚子|屁股|皮肤|痘|脱发).{0,12}(焦虑|自卑|丑|胖|不敢|崩溃|暴食|反弹)"),
    ("family_control", r"(父母|妈妈|爸爸|家里|原生家庭).{0,12}(控制|逼|骂|不让|不同意|道德绑架|窒息|否定)"),
    ("ordinary_self_mockery", r"(笑死|无语|离谱|尴尬|怪不好意思|嘴比脑子快|理想很丰满现实很骨感)"),
]


def normalize_label(x: Any) -> Optional[int]:
    if pd.isna(x):
        return None
    try:
        value = int(float(x))
        if value in LABEL_NAMES:
            return value
    except Exception:
        pass
    s = str(x).strip().upper()
    return TARGET_TO_CODE.get(s)


def normalize_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).replace("\u200b", "").replace("\ufeff", "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = URL_RE.sub(" ", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def compact_for_char_ngrams(text: str, keep_hashtags: bool) -> str:
    s = text
    if not keep_hashtags:
        s = HASHTAG_RE.sub(" ", s)
    s = re.sub(r"\s+", "", s)
    return s


def char_ngrams(text: str, min_n: int, max_n: int, keep_hashtags: bool) -> Set[str]:
    s = compact_for_char_ngrams(text, keep_hashtags=keep_hashtags)
    out: Set[str] = set()
    for n in range(min_n, max_n + 1):
        if len(s) < n:
            continue
        for i in range(len(s) - n + 1):
            term = s[i:i + n]
            if DROP_TERM_RE.match(term):
                continue
            if all(ch in STOP_CHARS for ch in term):
                continue
            if re.search(r"[\U00010000-\U0010ffff]", term):
                continue
            out.add(f"char:{term}")
    return out


def hashtag_terms(text: str) -> Set[str]:
    return {f"hash:{m.group(0).strip()}" for m in HASHTAG_RE.finditer(text) if len(m.group(0).strip()) >= 2}


def latin_terms(text: str) -> Set[str]:
    return {f"word:{m.group(0).lower()}" for m in LATIN_RE.finditer(text)}


def extract_terms(text: str, min_n: int, max_n: int, keep_hashtags_in_char: bool) -> Set[str]:
    terms = set()
    terms.update(char_ngrams(text, min_n=min_n, max_n=max_n, keep_hashtags=keep_hashtags_in_char))
    terms.update(hashtag_terms(text))
    terms.update(latin_terms(text))
    return terms


def display_term(term: str) -> Tuple[str, str]:
    if ":" in term:
        feature_type, value = term.split(":", 1)
        return feature_type, value
    return "unknown", term


def load_train(path: Path, limit: int = 0, balanced_limit_per_class: int = 0) -> pd.DataFrame:
    df = pd.read_excel(path)
    required = {"content", "class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    if "id" not in df.columns:
        df["id"] = [f"train_{i:05d}" for i in range(len(df))]
    df = df[["id", "content", "class"]].copy()
    df["content"] = df["content"].map(normalize_text)
    df["label_code"] = df["class"].map(normalize_label)
    df = df[df["content"].str.strip() != ""].copy()
    df = df[df["label_code"].isin([0, 1, 2])].copy()
    df["label_code"] = df["label_code"].astype(int)
    if balanced_limit_per_class > 0:
        df = (
            df.groupby("label_code", group_keys=False)
            .head(balanced_limit_per_class)
            .sort_index()
            .copy()
        )
    elif limit > 0:
        df = df.head(limit).copy()
    return df.reset_index(drop=True)


def build_doc_terms(df: pd.DataFrame, min_n: int, max_n: int, keep_hashtags_in_char: bool) -> List[Set[str]]:
    return [
        extract_terms(str(text), min_n=min_n, max_n=max_n, keep_hashtags_in_char=keep_hashtags_in_char)
        for text in df["content"].tolist()
    ]


def build_class_counts(doc_terms: Sequence[Set[str]], labels: Sequence[int]) -> Tuple[Dict[int, Counter], Counter, Dict[int, int]]:
    class_counts: Dict[int, Counter] = {0: Counter(), 1: Counter(), 2: Counter()}
    global_counts: Counter = Counter()
    class_doc_counts = {0: 0, 1: 0, 2: 0}
    for terms, label in zip(doc_terms, labels):
        class_doc_counts[int(label)] += 1
        class_counts[int(label)].update(terms)
        global_counts.update(terms)
    return class_counts, global_counts, class_doc_counts


def chi_square(a: int, b: int, c: int, d: int) -> float:
    n = a + b + c + d
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    if n == 0 or denom == 0:
        return 0.0
    return float(n * (a * d - b * c) ** 2 / denom)


def mutual_information_binary(a: int, b: int, c: int, d: int) -> float:
    n = a + b + c + d
    if n == 0:
        return 0.0
    table = [(a, a + b, a + c), (b, a + b, b + d), (c, c + d, a + c), (d, c + d, b + d)]
    mi = 0.0
    for cell, row_sum, col_sum in table:
        if cell <= 0 or row_sum <= 0 or col_sum <= 0:
            continue
        mi += (cell / n) * math.log((cell * n) / (row_sum * col_sum))
    return float(mi)


def log_odds_z(
    y_count: int,
    y_total: int,
    other_count: int,
    other_total: int,
    prior_count: float,
    prior_total: float,
) -> float:
    alpha_w = max(prior_count, 1e-9)
    alpha_rest = max(prior_total - prior_count, 1e-9)
    y_rest = max(y_total - y_count, 0)
    other_rest = max(other_total - other_count, 0)
    delta = math.log((y_count + alpha_w) / (y_rest + alpha_rest))
    delta -= math.log((other_count + alpha_w) / (other_rest + alpha_rest))
    var = 1.0 / (y_count + alpha_w) + 1.0 / (other_count + alpha_w)
    return float(delta / math.sqrt(var)) if var > 0 else 0.0


def context_rule_hint(label: str, term_value: str, purity: float, ambiguity_label: str) -> str:
    if label == "UP":
        if re.search(r"offer|录取|上岸|留学|面试|瘦|变美|身材|旅行|演唱会|隐藏款|限量", term_value, re.I):
            return "仅当帖主已获得/拥有/体验高光或稀缺资源时支持 UP；若是否定、失败、攻略、求问或第三方信息，转 DOWN/NEUTRAL。"
        return "需确认文本呈现帖主相对优势；普通喜欢、推荐、好看、日常体验不单独构成 UP。"
    if label == "DOWN":
        if re.search(r"offer|录取|上岸|留学|面试|变美|瘦", term_value, re.I):
            return "若高光词出现在没拿到、失败、被拒、焦虑、被限制语境中，支持 DOWN；成功展示则不支持 DOWN。"
        return "需确认帖主处境相对更糟、更受限、更低能动；普通负面语气或玩梗不单独构成 DOWN。"
    if label == "NEUTRAL":
        return "作为 neutralizer：若主要是教程、攻略、产品、第三方信息、普通偏好或玩梗，抑制 UP/DOWN。"
    if ambiguity_label:
        return f"跨标签高频，易与 {ambiguity_label} 混淆；放入 AMBIGUOUS 或要求上下文二次判定。"
    return "需人工复核上下文条件。"


def mine_feature_scores(
    doc_terms: Sequence[Set[str]],
    labels: Sequence[int],
    min_df: int,
    prior_strength: float,
) -> pd.DataFrame:
    class_counts, global_counts, class_doc_counts = build_class_counts(doc_terms, labels)
    n_docs = len(doc_terms)
    total_global = sum(global_counts.values())
    class_totals = {label: sum(class_counts[label].values()) for label in [0, 1, 2]}
    prior_total = float(prior_strength)

    rows: List[Dict[str, Any]] = []
    for term, total_df in global_counts.items():
        if total_df < min_df:
            continue
        counts = {label: int(class_counts[label][term]) for label in [0, 1, 2]}
        best_code = max([0, 1, 2], key=lambda c: counts[c])
        best_label = LABEL_NAMES[best_code]
        total = sum(counts.values())
        if total <= 0:
            continue
        purity = counts[best_code] / total
        other_counts = {label: counts[label] for label in [0, 1, 2] if label != best_code}
        ambiguity_code = max(other_counts, key=other_counts.get)
        ambiguity = other_counts[ambiguity_code]
        ambiguity_label = LABEL_NAMES[ambiguity_code] if ambiguity > 0 else ""
        feature_type, value = display_term(term)

        row: Dict[str, Any] = {
            "term": value,
            "feature_type": feature_type,
            "suggested_target": best_label,
            "count_UP": counts[0],
            "count_NEUTRAL": counts[1],
            "count_DOWN": counts[2],
            "total_docfreq": total,
            "purity": round(purity, 4),
            "coverage": counts[best_code],
            "ambiguity": ambiguity,
            "ambiguity_label": ambiguity_label,
        }

        max_abs_z = 0.0
        max_chi2 = 0.0
        max_mi = 0.0
        for code in [0, 1, 2]:
            label = LABEL_NAMES[code]
            y_count = counts[code]
            other_count = total - y_count
            prior_count = prior_strength * (global_counts[term] / total_global) if total_global else 1.0
            z = log_odds_z(
                y_count=y_count,
                y_total=class_totals[code],
                other_count=other_count,
                other_total=sum(class_totals.values()) - class_totals[code],
                prior_count=prior_count,
                prior_total=prior_total,
            )
            a = y_count
            b = other_count
            c = class_doc_counts[code] - y_count
            d = (n_docs - class_doc_counts[code]) - other_count
            chi2 = chi_square(a, b, c, d)
            mi = mutual_information_binary(a, b, c, d)
            row[f"log_odds_z_{label}"] = round(z, 4)
            row[f"chi2_{label}"] = round(chi2, 4)
            row[f"mi_{label}"] = round(mi, 6)
            max_abs_z = max(max_abs_z, abs(z))
            max_chi2 = max(max_chi2, chi2)
            max_mi = max(max_mi, mi)

        row["max_abs_log_odds_z"] = round(max_abs_z, 4)
        row["max_chi2"] = round(max_chi2, 4)
        row["max_mi"] = round(max_mi, 6)
        if purity >= 0.72 and counts[best_code] >= max(3, min_df) and ambiguity <= max(2, int(counts[best_code] * 0.35)):
            bucket = "core_cue"
        elif purity >= 0.55 and max_abs_z >= 1.5:
            bucket = "conditional_cue"
        else:
            bucket = "ambiguous_or_review"
        if best_label == "NEUTRAL" and purity >= 0.55:
            bucket = "neutralizer"
        row["suggested_bucket"] = bucket
        row["context_rule_hint"] = context_rule_hint(best_label, value, purity, ambiguity_label)
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["suggested_target", "suggested_bucket", "max_abs_log_odds_z", "max_chi2", "total_docfreq"],
        ascending=[True, True, False, False, False],
    ).reset_index(drop=True)


def add_optional_l1_logistic_scores(features_df: pd.DataFrame, doc_terms: Sequence[Set[str]], labels: Sequence[int], max_features: int) -> pd.DataFrame:
    if features_df.empty:
        return features_df
    try:
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
    except Exception:
        features_df["l1_coef_UP"] = ""
        features_df["l1_coef_NEUTRAL"] = ""
        features_df["l1_coef_DOWN"] = ""
        features_df["l1_available"] = False
        return features_df

    ranked = features_df.sort_values("max_abs_log_odds_z", ascending=False).head(max_features)
    vocab = {f"{row.feature_type}:{row.term}" for row in ranked.itertuples(index=False)}
    dicts = [{term: 1 for term in terms if term in vocab} for terms in doc_terms]
    vec = DictVectorizer(sparse=True)
    x = vec.fit_transform(dicts)
    y = np.asarray(labels, dtype=int)
    base_clf = LogisticRegression(penalty="l1", solver="liblinear", C=0.8, class_weight="balanced", max_iter=1000)
    clf = OneVsRestClassifier(base_clf)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message="Inconsistent values: penalty=.*")
        clf.fit(x, y)
    names = vec.get_feature_names_out()
    coef_map: Dict[str, Dict[int, float]] = defaultdict(dict)
    for estimator, code in zip(clf.estimators_, clf.classes_):
        for name, coef in zip(names, estimator.coef_[0]):
            if coef != 0:
                coef_map[name][int(code)] = float(coef)

    def get_coef(row: pd.Series, code: int) -> Any:
        key = f"{row['feature_type']}:{row['term']}"
        value = coef_map.get(key, {}).get(code)
        return round(value, 6) if value is not None else 0.0

    features_df["l1_coef_UP"] = features_df.apply(lambda r: get_coef(r, 0), axis=1)
    features_df["l1_coef_NEUTRAL"] = features_df.apply(lambda r: get_coef(r, 1), axis=1)
    features_df["l1_coef_DOWN"] = features_df.apply(lambda r: get_coef(r, 2), axis=1)
    features_df["l1_available"] = True
    return features_df


def stratified_folds(labels: Sequence[int], n_folds: int) -> List[List[int]]:
    buckets: Dict[int, List[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        buckets[int(label)].append(i)
    folds: List[List[int]] = [[] for _ in range(n_folds)]
    for bucket in buckets.values():
        for j, idx in enumerate(bucket):
            folds[j % n_folds].append(idx)
    return folds


def nb_predict(
    train_terms: Sequence[Set[str]],
    train_labels: Sequence[int],
    test_terms: Sequence[Set[str]],
    vocab: Set[str],
    alpha: float,
) -> List[int]:
    class_counts = {0: Counter(), 1: Counter(), 2: Counter()}
    class_docs = {0: 0, 1: 0, 2: 0}
    for terms, label in zip(train_terms, train_labels):
        label = int(label)
        class_docs[label] += 1
        class_counts[label].update(term for term in terms if term in vocab)
    total_docs = sum(class_docs.values())
    vocab_size = max(len(vocab), 1)
    preds: List[int] = []
    for terms in test_terms:
        present = terms & vocab
        scores: Dict[int, float] = {}
        for label in [0, 1, 2]:
            score = math.log((class_docs[label] + alpha) / (total_docs + 3 * alpha))
            denom = class_docs[label] + 2 * alpha
            for term in present:
                p = (class_counts[label][term] + alpha) / denom
                score += math.log(max(p, 1e-12))
            scores[label] = score
        preds.append(max(scores, key=scores.get))
    return preds


def hard_negative_mining(
    df: pd.DataFrame,
    doc_terms: Sequence[Set[str]],
    labels: Sequence[int],
    features_df: pd.DataFrame,
    n_folds: int,
    top_vocab: int,
) -> pd.DataFrame:
    if len(set(labels)) < 2:
        return pd.DataFrame()
    if len(df) < n_folds or features_df.empty:
        return pd.DataFrame()
    ranked = features_df.sort_values("max_abs_log_odds_z", ascending=False).head(top_vocab)
    vocab = {f"{row.feature_type}:{row.term}" for row in ranked.itertuples(index=False)}
    z_lookup: Dict[Tuple[int, str], float] = {}
    for row in ranked.itertuples(index=False):
        key = f"{row.feature_type}:{row.term}"
        z_lookup[(0, key)] = float(getattr(row, "log_odds_z_UP"))
        z_lookup[(1, key)] = float(getattr(row, "log_odds_z_NEUTRAL"))
        z_lookup[(2, key)] = float(getattr(row, "log_odds_z_DOWN"))

    folds = stratified_folds(labels, n_folds=n_folds)
    rows: List[Dict[str, Any]] = []
    all_indices = set(range(len(df)))
    for fold_id, test_idx in enumerate(folds):
        test_set = set(test_idx)
        train_idx = sorted(all_indices - test_set)
        preds = nb_predict(
            train_terms=[doc_terms[i] for i in train_idx],
            train_labels=[labels[i] for i in train_idx],
            test_terms=[doc_terms[i] for i in test_idx],
            vocab=vocab,
            alpha=0.5,
        )
        for idx, pred in zip(test_idx, preds):
            gold = int(labels[idx])
            if pred == gold:
                continue
            terms = doc_terms[idx] & vocab
            pred_terms = sorted(terms, key=lambda t: z_lookup.get((pred, t), 0.0), reverse=True)[:12]
            gold_terms = sorted(terms, key=lambda t: z_lookup.get((gold, t), 0.0), reverse=True)[:12]
            rows.append({
                "fold": fold_id,
                "id": df.iloc[idx]["id"],
                "gold_label": LABEL_NAMES[gold],
                "predicted_label": LABEL_NAMES[pred],
                "error_type": f"{LABEL_NAMES[gold]}->{LABEL_NAMES[pred]}",
                "content": df.iloc[idx]["content"],
                "pred_support_terms": "|".join(display_term(t)[1] for t in pred_terms),
                "gold_support_terms_present": "|".join(display_term(t)[1] for t in gold_terms),
            })
    return pd.DataFrame(rows)


def mine_negation_patterns(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for name, pattern in NEGATION_PATTERNS:
        regex = re.compile(pattern, re.I)
        class_counts = Counter()
        examples: Dict[int, List[str]] = defaultdict(list)
        for row in df.itertuples(index=False):
            text = str(row.content)
            matches = list(regex.finditer(text))
            if not matches:
                continue
            label = int(row.label_code)
            class_counts[label] += 1
            for m in matches[:2]:
                start = max(0, m.start() - 10)
                end = min(len(text), m.end() + 10)
                snippet = text[start:end].replace("\n", " ")
                if len(examples[label]) < 3:
                    examples[label].append(snippet)
        total = sum(class_counts.values())
        if total == 0:
            continue
        best = max([0, 1, 2], key=lambda c: class_counts[c])
        rows.append({
            "pattern_name": name,
            "regex": pattern,
            "suggested_target": LABEL_NAMES[best],
            "count_UP": class_counts[0],
            "count_NEUTRAL": class_counts[1],
            "count_DOWN": class_counts[2],
            "total_docfreq": total,
            "purity": round(class_counts[best] / total, 4),
            "example_UP": " || ".join(examples.get(0, [])),
            "example_NEUTRAL": " || ".join(examples.get(1, [])),
            "example_DOWN": " || ".join(examples.get(2, [])),
        })
    return pd.DataFrame(rows).sort_values(["suggested_target", "total_docfreq"], ascending=[True, False]) if rows else pd.DataFrame()


def char_jaccard(a: str, b: str) -> float:
    sa = set(a.lower())
    sb = set(b.lower())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def load_seed_cues(path: Optional[Path]) -> List[Tuple[str, str]]:
    if not path or not path.exists():
        return []
    df = pd.read_csv(path)
    if "cue" not in df.columns:
        return []
    seeds: List[Tuple[str, str]] = []
    for row in df.itertuples(index=False):
        cue = str(getattr(row, "cue", "")).strip()
        target = str(getattr(row, "target_label", "")).strip().upper() if "target_label" in df.columns else ""
        if cue and len(cue) >= 2:
            seeds.append((cue, target))
    return seeds


def seed_expansion(features_df: pd.DataFrame, seed_path: Optional[Path], min_similarity: float, max_rows: int) -> pd.DataFrame:
    seeds = load_seed_cues(seed_path)
    if not seeds or features_df.empty:
        return pd.DataFrame()
    candidates = features_df[
        (features_df["feature_type"].isin(["char", "hash", "word"]))
        & (features_df["purity"] >= 0.55)
    ].copy()
    rows: List[Dict[str, Any]] = []
    for seed, seed_target in seeds:
        for cand in candidates.itertuples(index=False):
            term = str(cand.term)
            if term == seed or len(term) < 2:
                continue
            sim = char_jaccard(seed, term)
            if sim < min_similarity:
                continue
            if seed_target and seed_target in TARGET_TO_CODE and str(cand.suggested_target) != LABEL_NAMES[TARGET_TO_CODE[seed_target]]:
                continue
            rows.append({
                "seed": seed,
                "seed_target": seed_target,
                "candidate": term,
                "feature_type": cand.feature_type,
                "suggested_target": cand.suggested_target,
                "similarity": round(sim, 4),
                "purity": cand.purity,
                "coverage": cand.coverage,
                "ambiguity": cand.ambiguity,
                "context_rule_hint": cand.context_rule_hint,
            })
            if len(rows) >= max_rows:
                break
        if len(rows) >= max_rows:
            break
    return pd.DataFrame(rows).sort_values(["suggested_target", "similarity", "coverage"], ascending=[True, False, False]) if rows else pd.DataFrame()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine train-only supervised lexicon candidates for XHS social comparison.")
    parser.add_argument("--train_path", type=Path, default=Path("/Users/jackie/Downloads/Yu EMNLP/XHS_SC_BERT/data/train.xlsx"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/lexicon/train_supervised_mining"))
    parser.add_argument("--limit", type=int, default=0, help="Debug limit. Default 0 uses all train rows.")
    parser.add_argument("--balanced_limit_per_class", type=int, default=0, help="Debug balanced sample size per class. Overrides --limit when > 0.")
    parser.add_argument("--min_char_n", type=int, default=2)
    parser.add_argument("--max_char_n", type=int, default=6)
    parser.add_argument("--min_df", type=int, default=5)
    parser.add_argument("--prior_strength", type=float, default=500.0)
    parser.add_argument("--keep_hashtags_in_char", action="store_true")
    parser.add_argument("--with_l1", action="store_true", help="Try L1 logistic coefficients if sklearn is installed.")
    parser.add_argument("--l1_max_features", type=int, default=5000)
    parser.add_argument("--hard_negative_folds", type=int, default=5)
    parser.add_argument("--hard_negative_top_vocab", type=int, default=3000)
    parser.add_argument("--seed_lexicon", type=Path, default=Path("outputs/lexicon/xhs_social_comparison_lexicon_train_augmented.csv"))
    parser.add_argument("--seed_min_similarity", type=float, default=0.55)
    parser.add_argument("--seed_max_rows", type=int, default=3000)
    args = parser.parse_args()

    df = load_train(args.train_path, limit=args.limit, balanced_limit_per_class=args.balanced_limit_per_class)
    labels = df["label_code"].astype(int).tolist()
    doc_terms = build_doc_terms(
        df,
        min_n=args.min_char_n,
        max_n=args.max_char_n,
        keep_hashtags_in_char=args.keep_hashtags_in_char,
    )

    features = mine_feature_scores(
        doc_terms=doc_terms,
        labels=labels,
        min_df=args.min_df,
        prior_strength=args.prior_strength,
    )
    if args.with_l1:
        features = add_optional_l1_logistic_scores(features, doc_terms, labels, max_features=args.l1_max_features)

    hard_negatives = hard_negative_mining(
        df=df,
        doc_terms=doc_terms,
        labels=labels,
        features_df=features,
        n_folds=max(2, args.hard_negative_folds),
        top_vocab=args.hard_negative_top_vocab,
    )
    negation_patterns = mine_negation_patterns(df)
    expansions = seed_expansion(
        features_df=features,
        seed_path=args.seed_lexicon,
        min_similarity=args.seed_min_similarity,
        max_rows=args.seed_max_rows,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.output_dir / "train_supervised_ngram_features.csv"
    hard_negative_path = args.output_dir / "train_cv_hard_negatives.csv"
    negation_path = args.output_dir / "train_negation_scope_patterns.csv"
    expansion_path = args.output_dir / "train_seed_expansion_candidates.csv"

    write_csv(features, feature_path)
    write_csv(hard_negatives, hard_negative_path)
    write_csv(negation_patterns, negation_path)
    write_csv(expansions, expansion_path)

    summary = {
        "train_path": str(args.train_path),
        "n_train": int(len(df)),
        "label_counts": {LABEL_NAMES[int(k)]: int(v) for k, v in df["label_code"].value_counts().sort_index().items()},
        "feature_rows": int(len(features)),
        "hard_negative_rows": int(len(hard_negatives)),
        "negation_pattern_rows": int(len(negation_patterns)),
        "seed_expansion_rows": int(len(expansions)),
        "outputs": {
            "features": str(feature_path),
            "hard_negatives": str(hard_negative_path),
            "negation_patterns": str(negation_path),
            "seed_expansion": str(expansion_path),
        },
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
