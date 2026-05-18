from pathlib import Path
import json
import re

import pandas as pd
from pypdf import PdfReader


ROOT = Path("/Users/jackie/Documents/Codex/2026-05-18/1-incom-social-comparison-orientation-scale")
DOWNLOADS = Path("/Users/jackie/Downloads/Yu EMNLP")
OUT = ROOT / "outputs" / "inspection"
OUT.mkdir(parents=True, exist_ok=True)


pdfs = {
    "xhs_score": DOWNLOADS / "xhs_score_paper.pdf",
    "caclp": DOWNLOADS / "8221_CACLP_Context_aware_Contr (2).pdf",
    "incom": ROOT / "sources" / "incom_gibbons_buunk_1999.pdf",
    "liwc2015": ROOT / "sources" / "liwc2015_development_manual.pdf",
    "upacs_dacs_appendix": ROOT / "sources" / "upacs_dacs_scale_appendix_haigazian.pdf",
    "upacs_dacs_frontiers": ROOT / "sources" / "upacs_dacs_german_validation_2024_frontiers.pdf",
}

focus_terms = [
    "lexicon", "dictionary", "prompt", "generation", "detection", "dissociation",
    "context", "contrast", "social comparison", "comparison", "upward", "downward",
    "Xiaohongshu", "RED", "score", "LIWC", "appearance"
]


def normalize(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_summary(path: Path, max_pages: int = 12):
    reader = PdfReader(str(path))
    page_count = len(reader.pages)
    pages = []
    for i in range(min(page_count, max_pages)):
        text = reader.pages[i].extract_text() or ""
        pages.append(f"\n\n--- page {i+1} ---\n{text}")
    head = normalize("".join(pages))

    hits = []
    for i, page in enumerate(reader.pages):
        text = normalize(page.extract_text() or "")
        lower = text.lower()
        if any(term.lower() in lower for term in focus_terms):
            snippets = []
            for term in focus_terms:
                idx = lower.find(term.lower())
                if idx >= 0:
                    snippets.append(text[max(0, idx - 260): idx + 520])
            hits.append({"page": i + 1, "snippets": snippets[:4]})
        if len(hits) >= 18:
            break
    return {"file": str(path), "pages": page_count, "head": head[:12000], "hits": hits}


def inspect_xlsx(path: Path):
    xls = pd.ExcelFile(path)
    sheets = {}
    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        sample = df.head(8).fillna("").astype(str).to_dict(orient="records")
        sheets[sheet] = {
            "shape": df.shape,
            "columns": [str(c) for c in df.columns],
            "sample": sample,
        }
    return {"file": str(path), "sheets": sheets}


result = {
    "pdfs": {name: extract_pdf_summary(path) for name, path in pdfs.items() if path.exists()},
    "xlsx": {
        name: inspect_xlsx(DOWNLOADS / f"{name}_full.xlsx")
        for name in ["DC", "NC", "UC"]
        if (DOWNLOADS / f"{name}_full.xlsx").exists()
    },
}

(OUT / "input_inspection.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({
    "pdfs": {k: {"pages": v["pages"], "hit_pages": [h["page"] for h in v["hits"][:8]]} for k, v in result["pdfs"].items()},
    "xlsx": {k: {s: meta["shape"] for s, meta in v["sheets"].items()} for k, v in result["xlsx"].items()},
}, ensure_ascii=False, indent=2))
