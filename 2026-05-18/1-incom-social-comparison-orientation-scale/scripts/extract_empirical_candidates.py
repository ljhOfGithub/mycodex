from pathlib import Path
from collections import Counter, defaultdict
import csv
import math
import re

import pandas as pd


ROOT = Path("/Users/jackie/Documents/Codex/2026-05-18/1-incom-social-comparison-orientation-scale")
DOWNLOADS = Path("/Users/jackie/Downloads/Yu EMNLP")
OUT = ROOT / "outputs" / "lexicon"
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "DOWN": DOWNLOADS / "DC_full.xlsx",
    "NEUTRAL": DOWNLOADS / "NC_full.xlsx",
    "UP": DOWNLOADS / "UC_full.xlsx",
}

STOP = set("的一是在不了和有就都也很还我你他她它们这那啊呀呢吗吧被把与及或")
DROP_PAT = re.compile(r"^[0-9a-zA-Z_#]+$")


def clean(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"#[^#\s]{1,40}", " ", text)
    text = re.sub(r"[\s\r\n\t]+", "", text)
    return text


def ngrams(text, min_n=2, max_n=6):
    text = clean(text)
    seen = set()
    for n in range(min_n, max_n + 1):
        for i in range(0, max(0, len(text) - n + 1)):
            g = text[i:i+n]
            if DROP_PAT.match(g):
                continue
            if all(ch in STOP for ch in g):
                continue
            if re.search(r"[\U00010000-\U0010ffff]", g):
                continue
            seen.add(g)
    return seen


class_counts = {label: Counter() for label in FILES}
doc_counts = {}
for label, path in FILES.items():
    df = pd.read_excel(path)
    texts = df["content"].fillna("").astype(str).tolist()
    doc_counts[label] = len(texts)
    for text in texts:
        class_counts[label].update(ngrams(text))

all_terms = set()
for c in class_counts.values():
    all_terms.update(c)

rows = []
for term in all_terms:
    counts = {label: class_counts[label][term] for label in FILES}
    total = sum(counts.values())
    if total < 6:
        continue
    best = max(counts, key=counts.get)
    share = counts[best] / total
    if share < 0.58:
        continue
    # Penalize ubiquitous fragments; favor concentrated but not one-off terms.
    score = share * math.log1p(total) * (counts[best] / doc_counts[best] * 1000)
    rows.append({
        "candidate": term,
        "best_label": best,
        "score": round(score, 4),
        "total_docs": total,
        "share_best": round(share, 3),
        "count_UP": counts["UP"],
        "count_DOWN": counts["DOWN"],
        "count_NEUTRAL": counts["NEUTRAL"],
    })

rows.sort(key=lambda r: (r["best_label"], -r["score"], -r["total_docs"], r["candidate"]))

path = OUT / "empirical_xhs_ngram_candidates.csv"
with path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"{len(rows)} candidates -> {path}")
for label in ["UP", "DOWN", "NEUTRAL"]:
    print("\n", label)
    shown = [r for r in rows if r["best_label"] == label][:20]
    for r in shown:
        print(r["candidate"], r["score"], r["count_UP"], r["count_DOWN"], r["count_NEUTRAL"])
