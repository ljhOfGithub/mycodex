#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate the current frame-based social-comparison lexicon with a global LLM prompt.

This script is adapted from:
    /Users/jackie/Downloads/Yu EMNLP/XHS_SC_BERT/run_xhs_lexicon_global_llm.py

It is designed for the lexicon produced in this workspace:
    outputs/lexicon/xhs_social_comparison_lexicon.csv

Input split format:
    id          optional, generated if absent
    content     required
    class       optional, required for metrics

Label codes:
    0 = UPWARD
    1 = NEUTRAL
    2 = DOWNWARD

Example VAL:
    export NEWAPI_API_KEY="your_key"
    python run_xhs_frame_lexicon_global_llm.py \
      --input "/Users/jackie/Downloads/Yu EMNLP/XHS_SC_BERT/data/val.xlsx" \
      --lexicon outputs/lexicon/xhs_social_comparison_lexicon.csv \
      --output outputs/frame_lexicon_val/val_framelex_top_all.csv \
      --base_url https://api.bianxie.ai \
      --model gpt-5 \
      --top_k_up 120 \
      --top_k_down 120 \
      --top_k_neu 70 \
      --top_k_ambiguous 46

Example TEST:
    export NEWAPI_API_KEY="your_key"
    python run_xhs_frame_lexicon_global_llm.py \
      --input "/Users/jackie/Downloads/Yu EMNLP/XHS_SC_BERT/data/test.xlsx" \
      --lexicon outputs/lexicon/xhs_social_comparison_lexicon.csv \
      --output outputs/frame_lexicon_test/test_framelex_top_all.csv \
      --base_url https://api.bianxie.ai \
      --model gpt-5 \
      --top_k_up 120 \
      --top_k_down 120 \
      --top_k_neu 70 \
      --top_k_ambiguous 46
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable: Iterable[Any], *args: Any, **kwargs: Any) -> Iterable[Any]:
        return iterable


DEFAULT_ENV_FILE = Path("config/api_keys.local.env")


LABEL_NAMES = {
    0: "UPWARD",
    1: "NEUTRAL",
    2: "DOWNWARD",
}

LABEL_TO_CODE = {
    "UPWARD": 0,
    "UP": 0,
    "NEUTRAL": 1,
    "NEU": 1,
    "DOWNWARD": 2,
    "DOWN": 2,
}

FRAME_LABEL_TO_CODE = {
    "UP": 0,
    "UPWARD": 0,
    "NEUTRAL": 1,
    "NEU": 1,
    "DOWN": 2,
    "DOWNWARD": 2,
}


def load_env_file(path: Path) -> Dict[str, str]:
    """
    Load a small KEY=VALUE env file without requiring python-dotenv.
    Existing process environment variables still take precedence in main().
    """
    if not path or not path.exists():
        return {}

    values: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


SYSTEM_PROMPT = """任务：仅根据【帖子文本】判断其最可能引发的比较方向，输出且仅输出一个标签（UPWARD/DOWNWARD/NEUTRAL）。

读者视角：18-24 岁活跃小红书用户的第一人称即时阅读反应。判断的是读者是否会把“我”和“帖主/文本中的人”放进比较关系，而不是作者意图、事实真伪或普通情绪正负。

标签定义：
- UPWARD：帖主/文本中的比较对象显得比我更好、更成功、更幸福、更有资源、更自由或更被认可，读者感觉对方相对更好。
- DOWNWARD：帖主显得比我更糟、更受限、更痛苦、更缺资源、更低能动性或更被控制，读者感觉自己相对更好。
- NEUTRAL：帖子不清晰邀请比较；帖主看起来与我相似，或文本没有明确自我-他人/读者-帖主定位。NEUTRAL 是实质性“无明显比较”，不是不确定。

分类线索（启发式，不是硬规则；以“是否邀请比较”+“方向”综合判断）：
A) 更可能为 DOWNWARD（向下比较：读者感觉自己相对更好）
- 叙事常呈现冲突/受挫/被指责/被控制：与父母/伴侣/他人争吵、被骂、被否定、被干涉。
- 低能动性/受害者叙事：大量被动句或“被...”结构、无力反抗、被夺走选择权。
- 强负面情绪与强化：难受/委屈/崩溃/窒息等 + 很/特别/超级等强化。
- 否定词密集：不、没、没有、别、从来不等。
- 报告动词/转述冲突：说、骂、问、逼等，突出指责对话。
- 明示“别人更好而我更差”：如“别人家的...都...我却...”。注意：这里“别人更好”是用来凸显帖主更差，从而让读者产生向下比较。

B) 更可能为 UPWARD（向上比较：读者感觉对方相对更好）
- 生活方式的“丰裕/理想/高光”框架：旅行、美食打卡、外貌管理、消费/购买体验、精致生活。
- 积极评价词与最高级/极值表达：好看、可爱、幸福、完美、最...、超级...、堪称...等。
- 适度感叹号与列举：用“！”增强兴奋感；用顿号/冒号/清单罗列美好事物或成就。
- 语气展示满足与拥有感：强调“我拥有/我体验到/我达成”的高位体验。

C) 更可能为 NEUTRAL（中性/无比较邀请）
- 信息型、说明型、教程型、天气/菜谱/客观建议/广告介绍为主。
- 低情绪或无个人立场；缺乏自我-他人定位；不暗示谁更好/更差。
- 即使出现积极/消极词，但不指向“我与他人/我与帖主”的相对地位。

注意：
1. 不要做普通情感分析。正面情绪不一定是 UPWARD，负面情绪不一定是 DOWNWARD。
2. 如果 cue 与整体语境冲突，以整体语境中的读者-帖主相对位置为准。
3. 只输出 JSON，不要输出解释文字。
"""


USER_PROMPT_TEMPLATE = """下面补充一组基于社会比较理论、XHS-SCoRE appendix cue-explicit rubric 和 XHS 语料构建的 frame-level lexicon。请把它当作上下文证据，不要机械匹配关键词。

【AMBIGUOUS comparison markers】
这些词提示可能存在 self-other / reader-poster 比较关系，但不能单独决定方向：
{ambiguous_frames}

【UPWARD frames】
这些 frame 通常表示帖主 advantaged / better off：
{up_frames}

【DOWNWARD frames】
这些 frame 通常表示帖主 disadvantaged / worse off：
{down_frames}

【NEUTRAL / neutralizer frames】
这些 frame 通常削弱社会比较判断，尤其是教程、产品、第三方信息、普通求推荐：
{neutral_frames}

判定步骤：
1. 先判断帖子是否邀请读者比较：是否存在自我-他人定位、读者-帖主定位、相对地位、资源、能动性、成就或生活质量差异。
2. 若邀请比较，再判断方向：
   - 对方/帖主整体更好、更高光、更有资源、更满足 -> UPWARD。
   - 帖主整体更糟、更受限、更痛苦、更缺资源、更低能动性 -> DOWNWARD。
3. 若主要是信息、教程、测评、新闻、榜单、工具说明、广告或求推荐，且没有帖主自身相对位置 -> NEUTRAL。
4. 若“别人更好”出现在帖主被比较、被贬低、被控制的叙事中，通常不是 UPWARD，而是凸显帖主更差，偏 DOWNWARD。
5. 若 UP 和 DOWN 线索同时出现，判断文本最终强调的是“已经获得优势/高光结果”还是“仍处于失败、受限、痛苦、缺资源”。

帖子：
{post_text}

仅输出 JSON：
{{"label":"UPWARD|DOWNWARD|NEUTRAL","dominant_frame":"frame name or NONE","comparison_relation":"explicit|implicit|absent","neutralizer_present":"yes|no"}}
"""


NLI_SYSTEM_PROMPT = """你是一个用于小红书社会比较检测的 NLI-style frame retriever。

任务：仅根据帖子文本，判断候选 frame 是否被当前上下文支持（entailed）、冲突（contradicted）或没有足够证据（neutral）。

你不是最终分类器。你的目标是为后续分类动态选择 context-relevant lexicon frames。

输出必须是 JSON，不要输出解释文字以外的内容。
"""


NLI_USER_PROMPT_TEMPLATE = """帖子：
{post_text}

候选 frames 来自词表 cue 命中。每个 frame 包含目标方向、命中的 cues、理论说明和语境规则。

{candidate_frames}

请判断每个候选 frame 是否被帖子上下文支持。

判断标准：
- entailed：帖子确实激活该 frame。例如 offer 真的指向帖主成就，而不是 offer 模板教程。
- contradicted：帖子上下文与该 frame 方向或功能冲突。例如“别人都成功我却失败”命中成功词，但整体是帖主更差。
- neutral：仅出现领域词或碎片，无法确认该 frame 被激活。

特别注意：
1. AMBIGUOUS comparison markers 只说明可能有比较关系，不单独决定 UP/DOWN。
2. NEUTRAL neutralizer frames 若被 entailed，应保留，因为它们能抑制误判。
3. “别人更好而我更差”应支持 DOWNWARD 的低能动/受挫 frame，而不是 UPWARD。

仅输出 JSON：
{{
  "comparison_relation": "explicit|implicit|absent",
  "selected_frames": [
    {{
      "target_label": "UP|DOWN|NEUTRAL|AMBIGUOUS",
      "frame": "frame_name",
      "nli_label": "entailed|neutral|contradicted",
      "confidence": 0.0,
      "reason": "short Chinese reason"
    }}
  ]
}}
"""


def normalize_base_url(url: str) -> str:
    url = url.rstrip("/")
    if url.endswith("/v1"):
        return url
    return url + "/v1"


def normalize_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    s = s.replace("\u200b", "")
    s = s.replace("\ufeff", "")
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


def parse_label(raw: str) -> Tuple[str, int, str, Dict[str, Any]]:
    if raw is None or str(raw).strip() == "":
        return "", -1, "empty_response", {}

    raw_clean = strip_code_fences(raw)
    try:
        obj = json.loads(raw_clean)
        label = str(obj.get("label", "")).strip().upper()
        if label in LABEL_TO_CODE:
            return LABEL_NAMES[LABEL_TO_CODE[label]], LABEL_TO_CODE[label], "", obj
        return "", -1, f"invalid_label:{label}", obj if isinstance(obj, dict) else {}
    except Exception as e:
        m = re.search(r"\b(UPWARD|DOWNWARD|NEUTRAL|UP|DOWN|NEU)\b", raw.upper())
        if m:
            label = m.group(1)
            return LABEL_NAMES[LABEL_TO_CODE[label]], LABEL_TO_CODE[label], "json_parse_failed_but_regex_found", {}
        return "", -1, f"json_parse_error:{repr(e)}", {}


def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    if "content" not in df.columns:
        raise ValueError(f"{path} must contain column: content")

    if "id" not in df.columns:
        df["id"] = [f"{path.stem}_{i:05d}" for i in range(len(df))]

    df["content"] = df["content"].fillna("").map(normalize_text)

    if "class" in df.columns:
        df["class"] = pd.to_numeric(df["class"], errors="coerce").astype("Int64")

    keep_cols = ["id", "content"] + (["class"] if "class" in df.columns else [])
    return df[keep_cols].copy()


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def load_frame_lexicon(
    lexicon_path: Path,
    min_weight: int,
    min_empirical_share: float,
    keep_empirical_mismatch: bool,
) -> pd.DataFrame:
    lex = pd.read_csv(lexicon_path)

    required = {"target_label", "frame", "cue"}
    missing = required - set(lex.columns)
    if missing:
        raise ValueError(f"Frame lexicon missing columns: {sorted(missing)}")

    lex = lex.copy()
    lex["target_label"] = lex["target_label"].astype(str).str.upper().str.strip()
    lex["frame"] = lex["frame"].astype(str).str.strip()
    lex["cue"] = lex["cue"].astype(str).str.strip()
    lex = lex[lex["cue"] != ""].copy()

    if "weight_1_3" not in lex.columns:
        lex["weight_1_3"] = 2
    lex["weight_1_3"] = lex["weight_1_3"].map(lambda x: _safe_int(x, 2))
    lex = lex[lex["weight_1_3"] >= min_weight].copy()

    for col in ["count_UP", "count_DOWN", "count_NEUTRAL"]:
        if col not in lex.columns:
            lex[col] = 0
        lex[col] = lex[col].map(_safe_int)

    if "empirical_share" not in lex.columns:
        lex["empirical_share"] = ""
    lex["empirical_share_num"] = lex["empirical_share"].map(lambda x: _safe_float(x, 0.0))

    if "empirical_label" not in lex.columns:
        lex["empirical_label"] = ""
    lex["empirical_label"] = lex["empirical_label"].fillna("").astype(str).str.upper().str.strip()

    if min_empirical_share > 0:
        no_empirical = lex["empirical_share_num"] <= 0
        enough_share = lex["empirical_share_num"] >= min_empirical_share
        lex = lex[no_empirical | enough_share].copy()

    if not keep_empirical_mismatch:
        def aligned(row: pd.Series) -> bool:
            empirical = str(row.get("empirical_label", "")).upper()
            target = str(row.get("target_label", "")).upper()
            if empirical == "" or empirical == "NAN":
                return True
            if target == "AMBIGUOUS":
                return True
            return empirical == target

        lex = lex[lex.apply(aligned, axis=1)].copy()

    return lex


def cue_sort_key(row: pd.Series) -> Tuple[float, int, int, str]:
    count_total = _safe_int(row.get("count_UP", 0)) + _safe_int(row.get("count_DOWN", 0)) + _safe_int(row.get("count_NEUTRAL", 0))
    return (
        -_safe_int(row.get("weight_1_3", 2)),
        -count_total,
        -_safe_float(row.get("empirical_share_num", 0.0)),
        str(row.get("cue", "")),
    )


def build_frame_blocks(
    lex: pd.DataFrame,
    target_label: str,
    top_k: int,
    max_chars: int,
    include_rules: bool,
) -> str:
    if top_k <= 0:
        return "无。"

    sub = lex[lex["target_label"] == target_label].copy()
    if sub.empty:
        return "无。"

    sub["_sort"] = sub.apply(cue_sort_key, axis=1)
    sub = sub.sort_values("_sort")

    lines: List[str] = []
    total_chars = 0

    for frame, g in sub.groupby("frame", sort=False):
        cues: List[str] = []
        for cue in g["cue"].tolist():
            cue = str(cue).strip()
            if cue and cue not in cues:
                cues.append(cue)
            if len(cues) >= top_k:
                break

        if not cues:
            continue

        rule = ""
        if include_rules and "context_rule" in g.columns:
            rules = [str(x).strip() for x in g["context_rule"].dropna().tolist() if str(x).strip()]
            if rules:
                rule = f" | rule: {rules[0]}"

        line = f"- {frame}: " + "、".join(cues) + rule
        add_len = len(line) + 1
        if max_chars > 0 and total_chars + add_len > max_chars:
            break
        lines.append(line)
        total_chars += add_len

    return "\n".join(lines) if lines else "无。"


def build_global_frames(args: argparse.Namespace, lex: pd.DataFrame) -> Dict[str, str]:
    return {
        "UP": build_frame_blocks(
            lex=lex,
            target_label="UP",
            top_k=args.top_k_up,
            max_chars=args.max_chars_per_label,
            include_rules=args.include_context_rules,
        ),
        "DOWN": build_frame_blocks(
            lex=lex,
            target_label="DOWN",
            top_k=args.top_k_down,
            max_chars=args.max_chars_per_label,
            include_rules=args.include_context_rules,
        ),
        "NEUTRAL": build_frame_blocks(
            lex=lex,
            target_label="NEUTRAL",
            top_k=args.top_k_neu,
            max_chars=args.max_chars_per_label,
            include_rules=args.include_context_rules,
        ),
        "AMBIGUOUS": build_frame_blocks(
            lex=lex,
            target_label="AMBIGUOUS",
            top_k=args.top_k_ambiguous,
            max_chars=args.max_chars_ambiguous,
            include_rules=args.include_context_rules,
        ),
    }


def cue_matches_text(cue: str, text: str) -> bool:
    cue = str(cue).strip()
    if not cue:
        return False
    if cue.startswith("re:"):
        try:
            return re.search(cue[3:], text) is not None
        except re.error:
            return False
    return cue in text


def build_nli_candidate_frames(
    lex: pd.DataFrame,
    post_text: str,
    max_candidate_frames: int,
    max_cues_per_frame: int,
    max_chars: int,
) -> Tuple[str, List[Tuple[str, str]]]:
    hits: List[Dict[str, Any]] = []
    for _, row in lex.iterrows():
        cue = str(row.get("cue", "")).strip()
        if cue_matches_text(cue, post_text):
            hits.append(row.to_dict())

    if not hits:
        return "无候选 frame：帖子没有直接命中当前 lexicon cues。", []

    hit_df = pd.DataFrame(hits)
    hit_df["_sort"] = hit_df.apply(cue_sort_key, axis=1)
    hit_df = hit_df.sort_values("_sort")

    grouped: List[Tuple[Tuple[str, str], pd.DataFrame, int]] = []
    for key, g in hit_df.groupby(["target_label", "frame"], sort=False):
        grouped.append((key, g, len(g)))

    grouped.sort(key=lambda item: (-item[2], str(item[0][0]), str(item[0][1])))

    lines: List[str] = []
    selected_keys: List[Tuple[str, str]] = []
    total_chars = 0

    for (target_label, frame), g, hit_count in grouped[:max_candidate_frames]:
        cues: List[str] = []
        for cue in g["cue"].tolist():
            cue = str(cue).strip()
            if cue and cue not in cues:
                cues.append(cue)
            if len(cues) >= max_cues_per_frame:
                break

        rationale = ""
        if "rationale" in g.columns:
            values = [str(x).strip() for x in g["rationale"].dropna().tolist() if str(x).strip()]
            rationale = values[0] if values else ""

        rule = ""
        if "context_rule" in g.columns:
            values = [str(x).strip() for x in g["context_rule"].dropna().tolist() if str(x).strip()]
            rule = values[0] if values else ""

        line = (
            f"- target_label={target_label}; frame={frame}; "
            f"matched_cues={','.join(cues)}; rationale={rationale}; context_rule={rule}"
        )
        add_len = len(line) + 1
        if max_chars > 0 and total_chars + add_len > max_chars:
            break
        lines.append(line)
        selected_keys.append((str(target_label), str(frame)))
        total_chars += add_len

    return "\n".join(lines) if lines else "无候选 frame。", selected_keys


def parse_nli_response(raw: str) -> Tuple[Dict[str, Any], str]:
    if raw is None or str(raw).strip() == "":
        return {}, "empty_response"

    raw_clean = strip_code_fences(raw)
    try:
        obj = json.loads(raw_clean)
        if isinstance(obj, dict):
            return obj, ""
        return {}, "nli_json_not_object"
    except Exception as e:
        return {}, f"nli_json_parse_error:{repr(e)}"


def call_nli_retrieval(
    client: Any,
    model: str,
    post_text: str,
    candidate_frames_text: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
    sleep_base: float,
    response_format_json: bool,
) -> Tuple[Dict[str, Any], str, str]:
    user_prompt = NLI_USER_PROMPT_TEMPLATE.format(
        post_text=post_text.strip(),
        candidate_frames=candidate_frames_text,
    )
    last_raw = ""
    last_error = ""

    for attempt in range(retries):
        try:
            kwargs: Dict[str, Any] = {
                "model": model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": NLI_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "timeout": timeout,
            }
            if response_format_json:
                kwargs["response_format"] = {"type": "json_object"}

            completion = client.chat.completions.create(**kwargs)
            raw = completion.choices[0].message.content or ""
            last_raw = raw
            parsed, parse_error = parse_nli_response(raw)
            if parsed:
                return parsed, raw, parse_error

            last_error = parse_error
            time.sleep(sleep_base * (attempt + 1))

        except Exception as e:
            last_error = repr(e)
            time.sleep(sleep_base * (attempt + 1))

    return {}, last_raw, last_error


def select_lexicon_by_nli(
    lex: pd.DataFrame,
    nli_obj: Dict[str, Any],
    threshold: float,
) -> Tuple[pd.DataFrame, List[str]]:
    selected = nli_obj.get("selected_frames", []) if isinstance(nli_obj, dict) else []
    selected_keys: List[Tuple[str, str]] = []
    selected_names: List[str] = []

    if not isinstance(selected, list):
        return lex.iloc[0:0].copy(), []

    for item in selected:
        if not isinstance(item, dict):
            continue
        nli_label = str(item.get("nli_label", "")).lower().strip()
        confidence = _safe_float(item.get("confidence", 0.0), 0.0)
        if nli_label != "entailed" or confidence < threshold:
            continue
        target_label = str(item.get("target_label", "")).upper().strip()
        frame = str(item.get("frame", "")).strip()
        if not target_label or not frame:
            continue
        selected_keys.append((target_label, frame))
        selected_names.append(f"{target_label}:{frame}:{confidence:.2f}")

    if not selected_keys:
        return lex.iloc[0:0].copy(), []

    mask = pd.Series(False, index=lex.index)
    for target_label, frame in selected_keys:
        mask = mask | ((lex["target_label"] == target_label) & (lex["frame"] == frame))
    return lex[mask].copy(), selected_names


def build_frames_from_selected_lex(
    args: argparse.Namespace,
    selected_lex: pd.DataFrame,
    fallback_frames: Dict[str, str],
    fallback_to_global: bool,
) -> Dict[str, str]:
    if selected_lex.empty:
        return fallback_frames if fallback_to_global else {
            "UP": "无。",
            "DOWN": "无。",
            "NEUTRAL": "无。",
            "AMBIGUOUS": "无。",
        }
    return build_global_frames(args, selected_lex)


def make_client(api_key: str, base_url: str) -> Any:
    if not api_key:
        raise RuntimeError("Missing API key. Set NEWAPI_API_KEY or OPENAI_API_KEY, or pass --api_key.")
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("Missing package: openai. Install it in the Python environment used to run this script.") from e
    return OpenAI(api_key=api_key, base_url=base_url)


def build_user_prompt(post_text: str, global_frames: Dict[str, str]) -> str:
    return USER_PROMPT_TEMPLATE.format(
        ambiguous_frames=global_frames.get("AMBIGUOUS", "无。"),
        up_frames=global_frames.get("UP", "无。"),
        down_frames=global_frames.get("DOWN", "无。"),
        neutral_frames=global_frames.get("NEUTRAL", "无。"),
        post_text=post_text.strip(),
    )


def classify_one(
    client: Any,
    model: str,
    post_text: str,
    global_frames: Dict[str, str],
    lex: Optional[pd.DataFrame],
    nli_args: Optional[argparse.Namespace],
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
    sleep_base: float,
    response_format_json: bool,
) -> Tuple[str, int, str, str, Dict[str, Any]]:
    active_frames = global_frames
    nli_raw = ""
    nli_error = ""
    nli_selected = ""

    if nli_args is not None and getattr(nli_args, "use_nli_retrieval", False) and lex is not None:
        candidate_frames_text, candidate_keys = build_nli_candidate_frames(
            lex=lex,
            post_text=post_text,
            max_candidate_frames=nli_args.nli_max_candidate_frames,
            max_cues_per_frame=nli_args.nli_max_cues_per_frame,
            max_chars=nli_args.nli_max_candidate_chars,
        )

        if candidate_keys:
            nli_model = nli_args.nli_model or model
            nli_obj, nli_raw, nli_error = call_nli_retrieval(
                client=client,
                model=nli_model,
                post_text=post_text,
                candidate_frames_text=candidate_frames_text,
                temperature=nli_args.nli_temperature,
                max_tokens=nli_args.nli_max_tokens,
                timeout=timeout,
                retries=retries,
                sleep_base=sleep_base,
                response_format_json=response_format_json,
            )
            selected_lex, selected_names = select_lexicon_by_nli(
                lex=lex,
                nli_obj=nli_obj,
                threshold=nli_args.nli_entail_threshold,
            )
            nli_selected = "|".join(selected_names)
            active_frames = build_frames_from_selected_lex(
                args=nli_args,
                selected_lex=selected_lex,
                fallback_frames=global_frames,
                fallback_to_global=nli_args.nli_fallback_global,
            )
        else:
            nli_error = "no_candidate_frames_from_lexicon_hits"
            if not nli_args.nli_fallback_global:
                active_frames = {
                    "UP": "无。",
                    "DOWN": "无。",
                    "NEUTRAL": "无。",
                    "AMBIGUOUS": "无。",
                }

    user_prompt = build_user_prompt(post_text=post_text, global_frames=active_frames)
    last_raw = ""
    last_error = ""

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

            completion = client.chat.completions.create(**kwargs)
            raw = completion.choices[0].message.content or ""
            last_raw = raw

            label, code, parse_error, parsed = parse_label(raw)
            if label:
                if isinstance(parsed, dict):
                    parsed["_nli_selected_frames"] = nli_selected
                    parsed["_nli_raw"] = nli_raw
                    parsed["_nli_error"] = nli_error
                return label, code, raw, parse_error, parsed

            last_error = parse_error
            time.sleep(sleep_base * (attempt + 1))

        except Exception as e:
            last_error = repr(e)
            time.sleep(sleep_base * (attempt + 1))

    return "", -1, last_raw, last_error, {
        "_nli_selected_frames": nli_selected,
        "_nli_raw": nli_raw,
        "_nli_error": nli_error,
    }


def classify_row_worker(
    row_payload: Dict[str, Any],
    api_key: str,
    base_url: str,
    model: str,
    global_frames: Dict[str, str],
    lex: pd.DataFrame,
    nli_args: argparse.Namespace,
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
    sleep_base: float,
    response_format_json: bool,
) -> Dict[str, Any]:
    client = make_client(api_key=api_key, base_url=base_url)
    label, code, raw, error, parsed = classify_one(
        client=client,
        model=model,
        post_text=str(row_payload["content"]),
        global_frames=global_frames,
        lex=lex,
        nli_args=nli_args,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        retries=retries,
        sleep_base=sleep_base,
        response_format_json=response_format_json,
    )

    return {
        "row_index": int(row_payload["row_index"]),
        "id": str(row_payload["id"]),
        "gt": row_payload.get("class", ""),
        "predicted": code if code in [0, 1, 2] else "",
        "predicted_label": label,
        "dominant_frame": parsed.get("dominant_frame", "") if isinstance(parsed, dict) else "",
        "comparison_relation": parsed.get("comparison_relation", "") if isinstance(parsed, dict) else "",
        "neutralizer_present": parsed.get("neutralizer_present", "") if isinstance(parsed, dict) else "",
        "nli_selected_frames": parsed.get("_nli_selected_frames", "") if isinstance(parsed, dict) else "",
        "nli_raw": parsed.get("_nli_raw", "") if isinstance(parsed, dict) else "",
        "nli_error": parsed.get("_nli_error", "") if isinstance(parsed, dict) else "",
        "raw": raw,
        "error": error,
    }


def compute_eval_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    valid = df[(df["gt"].notna()) & (df["predicted"].isin([0, 1, 2]))].copy()
    if valid.empty:
        return {"has_metrics": False, "reason": "No valid predictions or no gt."}

    y_true = valid["gt"].astype(int).to_numpy()
    y_pred = valid["predicted"].astype(int).to_numpy()
    cm = np.zeros((3, 3), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label in [0, 1, 2] and pred_label in [0, 1, 2]:
            cm[int(true_label), int(pred_label)] += 1

    recalls: List[float] = []
    f1s: List[float] = []
    for label in [0, 1, 2]:
        tp = float(cm[label, label])
        fn = float(cm[label, :].sum() - cm[label, label])
        fp = float(cm[:, label].sum() - cm[label, label])
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        recalls.append(recall)
        f1s.append(f1)

    up_mask = y_true == 0
    neu_mask = y_true == 1
    down_mask = y_true == 2

    return {
        "has_metrics": True,
        "n_eval": int(len(valid)),
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(np.mean(f1s)),
        "recall_upward": float(recalls[0]),
        "recall_neutral": float(recalls[1]),
        "recall_downward": float(recalls[2]),
        "predicted_upward_rate": float(np.mean(y_pred == 0)),
        "predicted_neutral_rate": float(np.mean(y_pred == 1)),
        "predicted_downward_rate": float(np.mean(y_pred == 2)),
        "up_to_neutral_rate": float(np.mean(y_pred[up_mask] == 1)) if np.any(up_mask) else 0.0,
        "down_to_neutral_rate": float(np.mean(y_pred[down_mask] == 1)) if np.any(down_mask) else 0.0,
        "neutral_to_up_rate": float(np.mean(y_pred[neu_mask] == 0)) if np.any(neu_mask) else 0.0,
        "neutral_to_down_rate": float(np.mean(y_pred[neu_mask] == 2)) if np.any(neu_mask) else 0.0,
        "up_to_down_rate": float(np.mean(y_pred[up_mask] == 2)) if np.any(up_mask) else 0.0,
        "down_to_up_rate": float(np.mean(y_pred[down_mask] == 0)) if np.any(down_mask) else 0.0,
        "confusion_matrix_labels_0_1_2": cm.tolist(),
    }


def save_metrics(metrics: Dict[str, Any], path: Path) -> None:
    metrics_path = path.with_suffix(".metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def is_successful_prediction_record(rec: Dict[str, Any]) -> bool:
    try:
        pred = rec.get("predicted", "")
        if pd.isna(pred):
            return False
        pred_int = int(float(pred))
        return pred_int in [0, 1, 2]
    except Exception:
        return False


def load_existing_results(output_path: Path, skip_existing_errors: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    if not output_path.exists():
        return [], {}

    try:
        existing_df = pd.read_csv(output_path)
    except Exception:
        return [], {}

    if existing_df.empty or "id" not in existing_df.columns:
        return [], {}

    existing_records: List[Dict[str, Any]] = []
    existing_by_id: Dict[str, Dict[str, Any]] = {}

    for fallback_idx, (_, row) in enumerate(existing_df.iterrows()):
        rec = row.to_dict()
        post_id = str(rec.get("id", ""))
        if not post_id or post_id.lower() == "nan":
            continue

        if "row_index" not in rec or pd.isna(rec.get("row_index")):
            rec["row_index"] = fallback_idx

        successful = is_successful_prediction_record(rec)
        if successful or skip_existing_errors:
            existing_records.append(rec)
            existing_by_id[post_id] = rec

    return existing_records, existing_by_id


def make_run_output_path(output: Path, run_idx: int, num_runs: int) -> Path:
    if num_runs <= 1:
        return output
    return output.with_name(f"{output.stem}_run{run_idx:02d}{output.suffix}")


def write_partial_results(results: List[Dict[str, Any]], run_output: Path) -> None:
    out_df = pd.DataFrame(results)
    if not out_df.empty and "row_index" in out_df.columns:
        out_df = out_df.sort_values("row_index").drop(columns=["row_index"], errors="ignore")
    out_df.to_csv(run_output, index=False, encoding="utf-8-sig")


def summarize_run_metrics(metrics_list: List[Dict[str, Any]], output: Path) -> None:
    usable = [m for m in metrics_list if m.get("has_metrics")]
    if not usable:
        summary = {
            "has_summary": False,
            "reason": "No usable run metrics.",
            "num_runs": len(metrics_list),
        }
        with open(output.with_suffix(".runs_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return

    metric_keys = [
        "accuracy",
        "macro_f1",
        "recall_upward",
        "recall_neutral",
        "recall_downward",
        "predicted_upward_rate",
        "predicted_neutral_rate",
        "predicted_downward_rate",
        "up_to_neutral_rate",
        "down_to_neutral_rate",
        "neutral_to_up_rate",
        "neutral_to_down_rate",
    ]

    rows = []
    for i, m in enumerate(usable, start=1):
        row = {"run": i}
        for k in metric_keys:
            row[k] = m.get(k)
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = output.with_suffix(".runs_summary.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary = {
        "has_summary": True,
        "num_runs_requested": len(metrics_list),
        "num_runs_with_metrics": len(usable),
        "runs_summary_csv": str(csv_path),
        "mean": {},
        "std": {},
    }
    for k in metric_keys:
        vals = pd.to_numeric(df[k], errors="coerce").dropna()
        summary["mean"][k] = float(vals.mean()) if len(vals) else None
        summary["std"][k] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

    with open(output.with_suffix(".runs_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


def run_one_experiment(
    args: argparse.Namespace,
    run_idx: int,
    run_output: Path,
    api_key: str,
    base_url: str,
    df: pd.DataFrame,
    global_frames: Dict[str, str],
    lex: pd.DataFrame,
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []

    existing_records, existing_by_id = load_existing_results(
        output_path=run_output,
        skip_existing_errors=args.skip_existing_errors,
    )
    if args.resume and existing_records:
        results.extend(existing_records)

    try:
        if args.concurrency <= 1:
            client = make_client(api_key=api_key, base_url=base_url)
            iterator = tqdm(df.iterrows(), total=len(df), desc=f"Classifying {args.input.name} run {run_idx}")

            for idx, row in iterator:
                post_id = str(row["id"])
                if args.resume and post_id in existing_by_id:
                    continue

                gt = int(row["class"]) if "class" in row and not pd.isna(row["class"]) else ""
                label, code, raw, error, parsed = classify_one(
                    client=client,
                    model=args.model,
                    post_text=str(row["content"]),
                    global_frames=global_frames,
                    lex=lex,
                    nli_args=args,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    retries=args.retries,
                    sleep_base=args.sleep_base,
                    response_format_json=not args.no_response_format_json,
                )

                results.append({
                    "row_index": int(idx),
                    "id": post_id,
                    "gt": gt,
                    "predicted": code if code in [0, 1, 2] else "",
                    "predicted_label": label,
                    "dominant_frame": parsed.get("dominant_frame", "") if isinstance(parsed, dict) else "",
                    "comparison_relation": parsed.get("comparison_relation", "") if isinstance(parsed, dict) else "",
                    "neutralizer_present": parsed.get("neutralizer_present", "") if isinstance(parsed, dict) else "",
                    "nli_selected_frames": parsed.get("_nli_selected_frames", "") if isinstance(parsed, dict) else "",
                    "nli_raw": parsed.get("_nli_raw", "") if isinstance(parsed, dict) else "",
                    "nli_error": parsed.get("_nli_error", "") if isinstance(parsed, dict) else "",
                    "raw": raw,
                    "error": error,
                })

                write_partial_results(results, run_output)

                if args.sleep_every > 0 and (idx + 1) % args.sleep_every == 0:
                    time.sleep(args.sleep_seconds)

        else:
            row_payloads: List[Dict[str, Any]] = []
            for idx, row in df.iterrows():
                post_id = str(row["id"])
                if args.resume and post_id in existing_by_id:
                    continue
                row_payloads.append({
                    "row_index": int(idx),
                    "id": post_id,
                    "content": str(row["content"]),
                    "class": int(row["class"]) if "class" in row and not pd.isna(row["class"]) else "",
                })

            completed = 0
            if row_payloads:
                with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                    future_to_row = {
                        executor.submit(
                            classify_row_worker,
                            row_payload,
                            api_key,
                            base_url,
                            args.model,
                            global_frames,
                            lex,
                            args,
                            args.temperature,
                            args.max_tokens,
                            args.timeout,
                            args.retries,
                            args.sleep_base,
                            not args.no_response_format_json,
                        ): row_payload
                        for row_payload in row_payloads
                    }

                    for future in tqdm(as_completed(future_to_row), total=len(future_to_row), desc=f"Classifying {args.input.name} run {run_idx}"):
                        row_payload = future_to_row[future]
                        try:
                            result = future.result()
                        except Exception as e:
                            result = {
                                "row_index": int(row_payload["row_index"]),
                                "id": row_payload.get("id", ""),
                                "gt": row_payload.get("class", ""),
                                "predicted": "",
                                "predicted_label": "",
                                "dominant_frame": "",
                                "comparison_relation": "",
                                "neutralizer_present": "",
                                "nli_selected_frames": "",
                                "nli_raw": "",
                                "nli_error": "",
                                "raw": "",
                                "error": f"future_error:{repr(e)}",
                            }

                        results.append(result)
                        completed += 1

                        if args.save_every > 0 and completed % args.save_every == 0:
                            write_partial_results(results, run_output)

                        if args.sleep_every > 0 and completed % args.sleep_every == 0:
                            time.sleep(args.sleep_seconds)

    finally:
        out_df = pd.DataFrame(results)
        if not out_df.empty and "row_index" in out_df.columns:
            out_df = out_df.sort_values("row_index").drop(columns=["row_index"])
        out_df.to_csv(run_output, index=False, encoding="utf-8-sig")

        metrics: Dict[str, Any] = {}
        if "gt" in out_df.columns and len(out_df) > 0:
            out_df["gt"] = pd.to_numeric(out_df["gt"], errors="coerce")
            out_df["predicted"] = pd.to_numeric(out_df["predicted"], errors="coerce")
            metrics = compute_eval_metrics(out_df)
            metrics.update({
                "run_idx": run_idx,
                "input": str(args.input),
                "lexicon": str(args.lexicon),
                "output": str(run_output),
                "model": args.model,
                "base_url": base_url,
                "top_k_up": args.top_k_up,
                "top_k_down": args.top_k_down,
                "top_k_neu": args.top_k_neu,
                "top_k_ambiguous": args.top_k_ambiguous,
                "min_weight": args.min_weight,
                "min_empirical_share": args.min_empirical_share,
                "keep_empirical_mismatch": args.keep_empirical_mismatch,
                "include_context_rules": args.include_context_rules,
                "use_nli_retrieval": args.use_nli_retrieval,
                "nli_model": args.nli_model or args.model,
                "nli_entail_threshold": args.nli_entail_threshold,
                "nli_fallback_global": args.nli_fallback_global,
                "nli_max_candidate_frames": args.nli_max_candidate_frames,
                "concurrency": args.concurrency,
                "resume": args.resume,
                "skip_existing_errors": args.skip_existing_errors,
                "lexicon_rows_after_filter": int(len(lex)),
                "frame_prompt_char_sizes": {k: len(v) for k, v in global_frames.items()},
            })
            save_metrics(metrics, run_output)
            print(json.dumps(metrics, indent=2, ensure_ascii=False))

        print(f"Saved predictions to {run_output}")
        return metrics


def write_prompt_preview(path: Path, global_frames: Dict[str, str]) -> None:
    preview = USER_PROMPT_TEMPLATE.format(
        ambiguous_frames=global_frames.get("AMBIGUOUS", "无。"),
        up_frames=global_frames.get("UP", "无。"),
        down_frames=global_frames.get("DOWN", "无。"),
        neutral_frames=global_frames.get("NEUTRAL", "无。"),
        post_text="【这里会替换为待分类的小红书帖子】",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SYSTEM_PROMPT + "\n\n" + preview, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frame-based XHS social comparison lexicon global LLM classifier.")

    parser.add_argument("--input", type=Path, required=True, help="Path to val.xlsx or test.xlsx")
    parser.add_argument("--lexicon", type=Path, default=Path("outputs/lexicon/xhs_social_comparison_lexicon.csv"))
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")

    parser.add_argument("--api_key", type=str, default="", help="API key. Prefer env file, NEWAPI_API_KEY, or OPENAI_API_KEY.")
    parser.add_argument("--env_file", type=Path, default=DEFAULT_ENV_FILE, help="Local KEY=VALUE file for API secrets/config.")
    parser.add_argument("--base_url", type=str, default="https://api.bianxie.ai", help="OpenAI-compatible base URL")
    parser.add_argument("--no_auto_v1", action="store_true", help="Do not append /v1 to base_url")
    parser.add_argument("--model", type=str, default="gpt-5")

    parser.add_argument("--top_k_up", type=int, default=120, help="Max cues per UP frame.")
    parser.add_argument("--top_k_down", type=int, default=120, help="Max cues per DOWN frame.")
    parser.add_argument("--top_k_neu", type=int, default=70, help="Max cues per NEUTRAL frame.")
    parser.add_argument("--top_k_ambiguous", type=int, default=46, help="Max cues per AMBIGUOUS frame.")
    parser.add_argument("--max_chars_per_label", type=int, default=10000)
    parser.add_argument("--max_chars_ambiguous", type=int, default=4000)

    parser.add_argument("--min_weight", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--min_empirical_share", type=float, default=0.0, help="Optional corpus precision-like filter; 0 keeps theory-only cues.")
    parser.add_argument("--keep_empirical_mismatch", action="store_true", help="Keep cues whose empirical majority label differs from target label.")
    parser.add_argument("--include_context_rules", action="store_true", help="Include one context rule per frame in the prompt.")

    parser.add_argument("--use_nli_retrieval", action="store_true", help="Enable NLI-style context-aware frame retrieval before final classification.")
    parser.add_argument("--nli_model", type=str, default="", help="Model for NLI retrieval. Defaults to --model.")
    parser.add_argument("--nli_temperature", type=float, default=0.0)
    parser.add_argument("--nli_max_tokens", type=int, default=1024)
    parser.add_argument("--nli_entail_threshold", type=float, default=0.55, help="Minimum confidence for an entailed frame to be kept.")
    parser.add_argument("--nli_max_candidate_frames", type=int, default=12, help="Max cue-hit frames sent to the NLI retriever per post.")
    parser.add_argument("--nli_max_cues_per_frame", type=int, default=8, help="Max matched cues shown per candidate frame.")
    parser.add_argument("--nli_max_candidate_chars", type=int, default=6000, help="Character budget for candidate frame list in the NLI prompt.")
    parser.add_argument("--nli_fallback_global", action="store_true", help="If NLI selects no frames, fall back to the global frame prompt instead of an empty frame prompt.")

    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep_base", type=float, default=1.0)
    parser.add_argument("--sleep_every", type=int, default=100)
    parser.add_argument("--sleep_seconds", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no_response_format_json", action="store_true", help="Disable response_format={json_object} for providers/models that reject it.")

    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_existing_errors", action="store_true")
    parser.add_argument("--prompt_preview", type=Path, default=Path("outputs/lexicon/frame_lexicon_prompt_preview.txt"))
    parser.add_argument("--only_write_prompt_preview", action="store_true", help="Build prompt preview and exit without calling the API.")

    args = parser.parse_args()

    env_values = load_env_file(args.env_file)

    api_key = (
        args.api_key
        or os.getenv("NEWAPI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or env_values.get("NEWAPI_API_KEY", "")
        or env_values.get("OPENAI_API_KEY", "")
        or env_values.get("API_KEY", "")
    )

    if args.base_url == parser.get_default("base_url"):
        base_url_raw = os.getenv("OPENAI_BASE_URL") or os.getenv("NEWAPI_BASE_URL") or env_values.get("OPENAI_BASE_URL", "") or env_values.get("NEWAPI_BASE_URL", "") or args.base_url
    else:
        base_url_raw = args.base_url
    base_url = base_url_raw.rstrip("/")
    if not args.no_auto_v1:
        base_url = normalize_base_url(base_url)

    if args.model == parser.get_default("model"):
        args.model = os.getenv("XHS_LLM_MODEL") or env_values.get("XHS_LLM_MODEL", "") or args.model

    df = load_split(args.input)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    lex = load_frame_lexicon(
        lexicon_path=args.lexicon,
        min_weight=args.min_weight,
        min_empirical_share=args.min_empirical_share,
        keep_empirical_mismatch=args.keep_empirical_mismatch,
    )
    global_frames = build_global_frames(args, lex)

    write_prompt_preview(args.prompt_preview, global_frames)

    print("Frame prompt character sizes:")
    print(json.dumps({k: len(v) for k, v in global_frames.items()}, indent=2, ensure_ascii=False))
    print(f"Lexicon rows after filtering: {len(lex)}")
    print(f"Prompt preview: {args.prompt_preview}")
    print(f"Env file: {args.env_file} ({'found' if args.env_file.exists() else 'not found'})")

    if args.only_write_prompt_preview:
        print("Exiting because --only_write_prompt_preview was set. No API calls made.")
        return

    print(f"Base URL: {base_url}")
    print(f"Model: {args.model}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Num runs: {args.num_runs}")
    print(f"Resume: {args.resume}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    metrics_list: List[Dict[str, Any]] = []
    for run_idx in range(1, args.num_runs + 1):
        run_output = make_run_output_path(args.output, run_idx=run_idx, num_runs=args.num_runs)
        print(f"\n========== RUN {run_idx}/{args.num_runs}: {run_output} ==========")
        metrics = run_one_experiment(
            args=args,
            run_idx=run_idx,
            run_output=run_output,
            api_key=api_key,
            base_url=base_url,
            df=df,
            global_frames=global_frames,
            lex=lex,
        )
        metrics_list.append(metrics)

    summarize_run_metrics(metrics_list, args.output)


if __name__ == "__main__":
    main()
