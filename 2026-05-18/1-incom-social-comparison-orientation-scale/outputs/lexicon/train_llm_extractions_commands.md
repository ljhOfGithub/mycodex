# LLM-assisted train lexicon extraction

These scripts are for building a more complete frame/cue lexicon from the gold-labeled TRAIN split. They do not modify the master lexicon automatically.

## Why

The current frame lexicon misses many XHS-native UP cues, such as:

- `solo trip`, `圆梦`, `演唱会`, `青旅`, `夜生活`
- `留学澳大利亚vlog`
- `丝芭面试`, `声乐专业`, `唱歌跳舞`, `路演`
- `隐藏款`, `拆盲盒`, `谷子`

This causes true UP posts to be over-neutralized.

## Step 1: pilot extraction

Run a small balanced pilot first:

```bash
python scripts/llm_extract_train_frame_cues.py \
  --train_path "/Users/jackie/Downloads/Yu EMNLP/XHS_SC_BERT/data/train.xlsx" \
  --output_jsonl outputs/lexicon/train_llm_extractions/pilot_300.jsonl \
  --balanced_limit_per_class 100 \
  --model gpt-4.1-nano \
  --base_url https://api.bianxie.ai \
  --concurrency 20 \
  --resume
```

Then consolidate:

```bash
python scripts/consolidate_train_llm_frame_cues.py \
  --input_jsonl outputs/lexicon/train_llm_extractions/pilot_300.jsonl \
  --raw_output outputs/lexicon/train_llm_extractions/pilot_300_raw.csv \
  --candidate_output outputs/lexicon/train_llm_extractions/pilot_300_candidates.csv \
  --min_docfreq 1
```

## Step 2: full train extraction

```bash
python scripts/llm_extract_train_frame_cues.py \
  --train_path "/Users/jackie/Downloads/Yu EMNLP/XHS_SC_BERT/data/train.xlsx" \
  --output_jsonl outputs/lexicon/train_llm_extractions/train_frame_cues.jsonl \
  --model gpt-4.1-nano \
  --base_url https://api.bianxie.ai \
  --concurrency 50 \
  --resume
```

Consolidate full output:

```bash
python scripts/consolidate_train_llm_frame_cues.py \
  --input_jsonl outputs/lexicon/train_llm_extractions/train_frame_cues.jsonl \
  --raw_output outputs/lexicon/train_llm_extractions/train_frame_cues_raw.csv \
  --candidate_output outputs/lexicon/train_llm_extractions/train_frame_cue_candidates.csv \
  --min_docfreq 2 \
  --min_confidence 0.5
```

## Recommended review

Sort `train_frame_cue_candidates.csv` by:

1. `target_label`
2. `llm_docfreq` descending
3. `mean_confidence` descending

Then manually merge stable candidates into `scripts/build_social_comparison_lexicon.py` or directly into a new master lexicon CSV.

## Step 3: merge TRAIN candidates into the master lexicon

Stable merge for validation/testing:

```bash
python scripts/merge_train_candidates_into_lexicon.py \
  --base_lexicon outputs/lexicon/xhs_social_comparison_lexicon.csv \
  --train_candidates outputs/lexicon/train_llm_extractions/train_frame_cue_candidates_stable.csv \
  --output outputs/lexicon/xhs_social_comparison_lexicon_train_augmented.csv \
  --review_output outputs/lexicon/xhs_social_comparison_lexicon_train_augmented_review.csv \
  --min_docfreq 2 \
  --min_confidence 0.5
```

Maximum-recall merge for ablation only:

```bash
python scripts/merge_train_candidates_into_lexicon.py \
  --base_lexicon outputs/lexicon/xhs_social_comparison_lexicon.csv \
  --train_candidates outputs/lexicon/train_llm_extractions/train_frame_cue_candidates.csv \
  --output outputs/lexicon/xhs_social_comparison_lexicon_train_augmented_recall.csv \
  --review_output outputs/lexicon/xhs_social_comparison_lexicon_train_augmented_recall_review.csv \
  --include_singletons \
  --min_confidence 0.0
```

Use `xhs_social_comparison_lexicon_train_augmented.csv` on VAL first. Only after choosing parameters on VAL should you run TEST.
