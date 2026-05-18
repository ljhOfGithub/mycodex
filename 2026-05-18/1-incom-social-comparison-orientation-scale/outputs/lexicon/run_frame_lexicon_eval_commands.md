# Frame lexicon evaluation commands

The evaluation script is:

`run_xhs_frame_lexicon_global_llm.py`

It follows the structure of `XHS_SC_BERT/run_xhs_lexicon_global_llm.py`, but reads the current frame lexicon:

`outputs/lexicon/xhs_social_comparison_lexicon.csv`

## Preview prompt only

This writes the global prompt preview and makes no API calls:

```bash
python run_xhs_frame_lexicon_global_llm.py \
  --input "/Users/jackie/Downloads/Yu EMNLP/XHS_SC_BERT/data/val.xlsx" \
  --lexicon outputs/lexicon/xhs_social_comparison_lexicon.csv \
  --output outputs/frame_lexicon_val/val_framelex_preview.csv \
  --only_write_prompt_preview \
  --include_context_rules
```

## VAL run

```bash
# First copy config/api_keys.example.env to config/api_keys.local.env,
# then put your real key in NEWAPI_API_KEY.

python run_xhs_frame_lexicon_global_llm.py \
  --input "/Users/jackie/Downloads/Yu EMNLP/XHS_SC_BERT/data/val.xlsx" \
  --lexicon outputs/lexicon/xhs_social_comparison_lexicon.csv \
  --output outputs/frame_lexicon_val/val_framelex_top_all.csv \
  --base_url https://api.bianxie.ai \
  --model gpt-5 \
  --top_k_up 120 \
  --top_k_down 120 \
  --top_k_neu 70 \
  --top_k_ambiguous 46 \
  --include_context_rules \
  --concurrency 5
```

## TEST run

```bash
# Uses config/api_keys.local.env by default.

python run_xhs_frame_lexicon_global_llm.py \
  --input "/Users/jackie/Downloads/Yu EMNLP/XHS_SC_BERT/data/test.xlsx" \
  --lexicon outputs/lexicon/xhs_social_comparison_lexicon.csv \
  --output outputs/frame_lexicon_test/test_framelex_top_all.csv \
  --base_url https://api.bianxie.ai \
  --model gpt-5 \
  --top_k_up 120 \
  --top_k_down 120 \
  --top_k_neu 70 \
  --top_k_ambiguous 46 \
  --include_context_rules \
  --concurrency 5
```

## Useful ablations

NLI context-aware retrieval:

```bash
python run_xhs_frame_lexicon_global_llm.py \
  --input "/Users/jackie/Downloads/Yu EMNLP/XHS_SC_BERT/data/val.xlsx" \
  --lexicon outputs/lexicon/xhs_social_comparison_lexicon.csv \
  --output outputs/frame_lexicon_val/val_framelex_nli.csv \
  --base_url https://api.bianxie.ai \
  --model gpt-5 \
  --use_nli_retrieval \
  --nli_model gpt-5 \
  --nli_entail_threshold 0.55 \
  --nli_max_candidate_frames 12 \
  --nli_max_cues_per_frame 8 \
  --include_context_rules
```

NLI retrieval with global fallback:

```bash
python run_xhs_frame_lexicon_global_llm.py \
  --input "/Users/jackie/Downloads/Yu EMNLP/XHS_SC_BERT/data/val.xlsx" \
  --lexicon outputs/lexicon/xhs_social_comparison_lexicon.csv \
  --output outputs/frame_lexicon_val/val_framelex_nli_fallback.csv \
  --base_url https://api.bianxie.ai \
  --model gpt-5 \
  --use_nli_retrieval \
  --nli_fallback_global \
  --include_context_rules
```

No neutralizer cues:

```bash
python run_xhs_frame_lexicon_global_llm.py \
  --input "/Users/jackie/Downloads/Yu EMNLP/XHS_SC_BERT/data/val.xlsx" \
  --lexicon outputs/lexicon/xhs_social_comparison_lexicon.csv \
  --output outputs/frame_lexicon_val/val_framelex_no_neu.csv \
  --base_url https://api.bianxie.ai \
  --model gpt-5 \
  --top_k_up 120 \
  --top_k_down 120 \
  --top_k_neu 0 \
  --top_k_ambiguous 46 \
  --include_context_rules
```

High-confidence only:

```bash
python run_xhs_frame_lexicon_global_llm.py \
  --input "/Users/jackie/Downloads/Yu EMNLP/XHS_SC_BERT/data/val.xlsx" \
  --lexicon outputs/lexicon/xhs_social_comparison_lexicon.csv \
  --output outputs/frame_lexicon_val/val_framelex_weight3.csv \
  --base_url https://api.bianxie.ai \
  --model gpt-5 \
  --min_weight 3 \
  --top_k_up 120 \
  --top_k_down 120 \
  --top_k_neu 70 \
  --top_k_ambiguous 46 \
  --include_context_rules
```

Repeated runs:

```bash
python run_xhs_frame_lexicon_global_llm.py \
  --input "/Users/jackie/Downloads/Yu EMNLP/XHS_SC_BERT/data/val.xlsx" \
  --lexicon outputs/lexicon/xhs_social_comparison_lexicon.csv \
  --output outputs/frame_lexicon_val/val_framelex_3runs.csv \
  --base_url https://api.bianxie.ai \
  --model gpt-5 \
  --num_runs 3 \
  --include_context_rules
```
