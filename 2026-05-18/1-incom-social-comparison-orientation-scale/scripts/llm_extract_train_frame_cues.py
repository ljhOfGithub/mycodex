#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM-assisted extraction of frame-level social-comparison cues from TRAIN.

This script does not classify posts. It uses the gold label in train.xlsx and asks
an OpenAI-compatible LLM to extract reusable frame/cue candidates that explain
that gold label. It writes JSONL incrementally, so it can be resumed safely.

Example:
    python scripts/llm_extract_train_frame_cues.py \
      --train_path "/Users/jackie/Downloads/Yu EMNLP/XHS_SC_BERT/data/train.xlsx" \
      --output_jsonl outputs/lexicon/train_llm_extractions/train_frame_cues.jsonl \
      --model gpt-4.1-nano \
      --base_url https://api.bianxie.ai \
      --concurrency 20 \
      --resume

For a quick pilot:
    python scripts/llm_extract_train_frame_cues.py \
      --train_path "/Users/jackie/Downloads/Yu EMNLP/XHS_SC_BERT/data/train.xlsx" \
      --output_jsonl outputs/lexicon/train_llm_extractions/pilot_300.jsonl \
      --balanced_limit_per_class 100 \
      --concurrency 10
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import pandas as pd


DEFAULT_ENV_FILE = Path("config/api_keys.local.env")
LABEL_NAMES = {0: "UPWARD", 1: "NEUTRAL", 2: "DOWNWARD"}


SYSTEM_PROMPT = """你是 XHS-SCoRE 社会比较 frame/cue 抽取助手。

任务：给定一条小红书帖子及其人工 gold label，抽取能够解释该 gold label 的可复用 lexicon candidates。

你不是分类器，不要推翻 gold label。你要帮助研究者从训练集中查漏补缺，尤其补足小红书 native 的隐性 UP/DOWN/NEUTRAL frame。

标签定义：
- UPWARD：帖子把帖主/文本中的比较对象呈现为更好、更成功、更幸福、更有资源、更自由、更被认可，读者可能感觉对方相对更好。
- DOWNWARD：帖子把帖主呈现为更糟、更受限、更痛苦、更缺资源、更低能动性、更被控制，读者可能感觉自己相对更好。
- NEUTRAL：帖子没有清晰读者-帖主相对位置，通常是信息、教程、产品、新闻、广告、第三方榜单、普通求推荐。

抽取原则：
1. cue 必须来自原帖，或是对原帖中反复出现模式的忠实短语概括；不要凭空造词。
2. 优先抽取能说明 reader-poster relative standing 的词、短语、叙事框架、话题域和语用模式。
3. UPWARD 不要求显式比较词。旅行/美食/演唱会/留学/港澳台海外移动性/圆梦/人生照片/隐藏款/稀缺机会/外貌优势/被认可/高光体验/拥有感，都可能是 UP cue。
4. DOWNWARD 关注冲突、被控制、低能动性、失败、受限、被否定、负面转述、“别人更好而我更差”。
5. NEUTRAL 关注信息型、教程型、测评、新闻、榜单、广告、产品介绍、客观建议、无个人立场。
6. 不要只抽情绪词；情绪词只能作为 supporting cue。
7. 输出必须是 JSON。
"""


USER_PROMPT_TEMPLATE = """gold_label: {gold_label}
post_id: {post_id}

帖子文本：
{post_text}

请输出 JSON：
{{
  "post_id": "{post_id}",
  "gold_label": "{gold_label}",
  "dominant_frames": [
    {{
      "target_label": "UP|DOWN|NEUTRAL|AMBIGUOUS",
      "frame": "snake_case_frame_name",
      "frame_description": "short Chinese description",
      "evidence": ["原帖证据1", "原帖证据2"],
      "confidence": 0.0
    }}
  ],
  "candidate_cues": [
    {{
      "target_label": "UP|DOWN|NEUTRAL|AMBIGUOUS",
      "frame": "snake_case_frame_name",
      "cue": "2-12字中文短语或短英文片段",
      "cue_type": "domain_frame|relational_marker|relative_standing|neutralizer_frame|affect_supporting|social_reward|style_marker",
      "rationale": "为什么这个 cue/frame 支持 gold_label",
      "context_rule": "什么时候不能机械使用",
      "weight_1_3": 1,
      "confidence": 0.0
    }}
  ],
  "negative_cues": [
    {{
      "cue": "看似相关但不应机械用于该标签的词",
      "reason": "为什么"
    }}
  ],
  "summary": "一句话说明这条帖子为什么是 gold_label"
}}

硬性格式要求：
- target_label 只能写 UP、DOWN、NEUTRAL、AMBIGUOUS 之一。
- cue_type 才可以写 relational_marker、domain_frame、neutralizer_frame 等。
- 不要把 relational_marker、domain_frame、appearance 等写进 target_label。
"""


def load_env_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    values: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def normalize_base_url(url: str) -> str:
    url = url.rstrip("/")
    return url if url.endswith("/v1") else url + "/v1"


def normalize_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).replace("\u200b", "").replace("\ufeff", "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t[3:].strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
        if t.endswith("```"):
            t = t[:-3].strip()
    return t.strip()


def parse_json(raw: str) -> Tuple[Dict[str, Any], str]:
    if not raw or not raw.strip():
        return {}, "empty_response"
    try:
        obj = json.loads(strip_code_fences(raw))
        if isinstance(obj, dict):
            return obj, ""
        return {}, "json_not_object"
    except Exception as e:
        return {}, f"json_parse_error:{repr(e)}"


def make_client(api_key: str, base_url: str) -> Any:
    if not api_key:
        raise RuntimeError("Missing API key. Fill config/api_keys.local.env or set NEWAPI_API_KEY.")
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("Missing package: openai. Install it in the environment used to run this script.") from e
    return OpenAI(api_key=api_key, base_url=base_url)


def load_train(path: Path, balanced_limit_per_class: int, limit: int) -> pd.DataFrame:
    df = pd.read_excel(path)
    required = {"content", "class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"train file missing columns: {sorted(missing)}")
    if "id" not in df.columns:
        df["id"] = [f"train_{i:05d}" for i in range(len(df))]

    df = df[["id", "content", "class"]].copy()
    df["content"] = df["content"].map(normalize_text)
    df["class"] = pd.to_numeric(df["class"], errors="coerce").astype("Int64")
    df = df[df["content"].str.strip() != ""].copy()
    df = df[df["class"].isin([0, 1, 2])].copy()

    if balanced_limit_per_class > 0:
        df = (
            df.groupby("class", group_keys=False)
            .head(balanced_limit_per_class)
            .sort_index()
            .copy()
        )
    elif limit > 0:
        df = df.head(limit).copy()
    return df


def load_done_ids(path: Path) -> Set[str]:
    done: Set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                post_id = obj.get("id")
                if post_id:
                    done.add(str(post_id))
            except Exception:
                continue
    return done


def call_one(
    client: Any,
    model: str,
    row: Dict[str, Any],
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
    sleep_base: float,
    response_format_json: bool,
) -> Dict[str, Any]:
    post_id = str(row["id"])
    label_code = int(row["class"])
    gold_label = LABEL_NAMES[label_code]
    post_text = str(row["content"])
    user_prompt = USER_PROMPT_TEMPLATE.format(
        post_id=post_id,
        gold_label=gold_label,
        post_text=post_text,
    )

    last_raw = ""
    last_error = ""
    parsed: Dict[str, Any] = {}

    for attempt in range(retries):
        try:
            kwargs: Dict[str, Any] = {
                "model": model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "timeout": timeout,
            }
            if response_format_json:
                kwargs["response_format"] = {"type": "json_object"}

            resp = client.chat.completions.create(**kwargs)
            last_raw = resp.choices[0].message.content or ""
            parsed, parse_error = parse_json(last_raw)
            if parsed:
                return {
                    "id": post_id,
                    "class": label_code,
                    "gold_label": gold_label,
                    "content": post_text,
                    "parsed": parsed,
                    "raw": last_raw,
                    "error": parse_error,
                }
            last_error = parse_error
        except Exception as e:
            last_error = repr(e)
        time.sleep(sleep_base * (attempt + 1))

    return {
        "id": post_id,
        "class": label_code,
        "gold_label": gold_label,
        "content": post_text,
        "parsed": parsed,
        "raw": last_raw,
        "error": last_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frame/cue candidates from train.xlsx with an LLM API.")
    parser.add_argument("--train_path", type=Path, default=Path("/Users/jackie/Downloads/Yu EMNLP/XHS_SC_BERT/data/train.xlsx"))
    parser.add_argument("--output_jsonl", type=Path, default=Path("outputs/lexicon/train_llm_extractions/train_frame_cues.jsonl"))
    parser.add_argument("--env_file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--base_url", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1400)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep_base", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--balanced_limit_per_class", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_response_format_json", action="store_true")
    args = parser.parse_args()

    env = load_env_file(args.env_file)
    api_key = args.api_key or os.getenv("NEWAPI_API_KEY") or os.getenv("OPENAI_API_KEY") or env.get("NEWAPI_API_KEY", "") or env.get("OPENAI_API_KEY", "")
    base_url = args.base_url or os.getenv("NEWAPI_BASE_URL") or os.getenv("OPENAI_BASE_URL") or env.get("NEWAPI_BASE_URL", "") or env.get("OPENAI_BASE_URL", "") or "https://api.bianxie.ai"
    model = args.model or os.getenv("XHS_LLM_MODEL") or env.get("XHS_LLM_MODEL", "") or "gpt-4.1-nano"
    base_url = normalize_base_url(base_url)

    df = load_train(args.train_path, args.balanced_limit_per_class, args.limit)
    done = load_done_ids(args.output_jsonl) if args.resume else set()
    if done:
        df = df[~df["id"].astype(str).isin(done)].copy()

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    client = make_client(api_key, base_url)
    lock = threading.Lock()
    written = 0

    print(json.dumps({
        "train_path": str(args.train_path),
        "output_jsonl": str(args.output_jsonl),
        "model": model,
        "base_url": base_url,
        "rows_to_process": int(len(df)),
        "resume_done": len(done),
        "concurrency": args.concurrency,
    }, ensure_ascii=False, indent=2))

    rows = df.to_dict(orient="records")
    mode = "a" if args.resume else "w"
    with args.output_jsonl.open(mode, encoding="utf-8") as out:
        if args.concurrency <= 1:
            for row in rows:
                rec = call_one(client, model, row, args.temperature, args.max_tokens, args.timeout, args.retries, args.sleep_base, not args.no_response_format_json)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                written += 1
                if written % args.save_every == 0:
                    print(f"written={written}")
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                futures = [
                    ex.submit(call_one, client, model, row, args.temperature, args.max_tokens, args.timeout, args.retries, args.sleep_base, not args.no_response_format_json)
                    for row in rows
                ]
                for fut in as_completed(futures):
                    rec = fut.result()
                    with lock:
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        out.flush()
                        written += 1
                        if written % args.save_every == 0:
                            print(f"written={written}")

    print(f"Done. wrote={written} -> {args.output_jsonl}")


if __name__ == "__main__":
    main()
