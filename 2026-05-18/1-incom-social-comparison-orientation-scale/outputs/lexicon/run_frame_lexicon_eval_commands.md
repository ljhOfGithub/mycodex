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
  --model gpt-4.1-nano \
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
  --model gpt-4.1-nano \
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
  --model gpt-4.1-nano \
  --use_nli_retrieval \
  --nli_model gpt-4.1-nano \
  --nli_entail_threshold 0.55 \
  --nli_max_candidate_frames 12 \
  --nli_max_cues_per_frame 8 \
  --include_context_rules
```

Recommended anti-neutralization run after the Appendix B.5 prompt update:

```bash
python run_xhs_frame_lexicon_global_llm.py \
  --input "/Users/jackie/Downloads/Yu EMNLP/XHS_SC_BERT/data/test.xlsx" \
  --lexicon outputs/lexicon/xhs_social_comparison_lexicon.csv \
  --output outputs/frame_lexicon_test/test_framelex_nli_broad_upfix.csv \
  --base_url https://api.bianxie.ai \
  --model gpt-4.1-nano \
  --use_nli_retrieval \
  --nli_model gpt-4.1-nano \
  --nli_entail_threshold 0.55 \
  --nli_max_candidate_frames 12 \
  --nli_max_cues_per_frame 8 \
  --nli_include_broad_hypotheses \
  --nli_fallback_global \
  --top_k_up 120 \
  --top_k_down 120 \
  --top_k_neu 70 \
  --top_k_ambiguous 46 \
  --include_context_rules \
  --concurrency 100
```

If budget allows, use a stronger NLI retriever while keeping the final classifier small:

```bash
python run_xhs_frame_lexicon_global_llm.py \
  --input "/Users/jackie/Downloads/Yu EMNLP/XHS_SC_BERT/data/test.xlsx" \
  --lexicon outputs/lexicon/xhs_social_comparison_lexicon.csv \
  --output outputs/frame_lexicon_test/test_framelex_nli_gpt5_retriever.csv \
  --base_url https://api.bianxie.ai \
  --model gpt-4.1-nano \
  --use_nli_retrieval \
  --nli_model gpt-5 \
  --nli_entail_threshold 0.55 \
  --nli_include_broad_hypotheses \
  --nli_fallback_global \
  --top_k_neu 70 \
  --include_context_rules \
  --concurrency 50
```

## Extract missing cues from UP -> NEUTRAL errors

Use this after a run has produced a prediction CSV. It asks an LLM to analyze
gold UPWARD / predicted NEUTRAL errors and propose missing frame-level lexicon
candidates for manual review.

If the prediction CSV has a `content` column:

```bash
python scripts/extract_llm_error_cues.py \
  --predictions outputs/frame_lexicon_test/test_framelex_nli_broad_upfix.csv \
  --output_dir outputs/lexicon/error_cue_extraction/up_to_neu \
  --gold_label 0 \
  --pred_label 1 \
  --limit 200 \
  --model gpt-4.1-nano
```

If the prediction CSV does not have a `content` column, merge text from the test
file:

```bash
python scripts/extract_llm_error_cues.py \
  --predictions outputs/frame_lexicon_test/test_framelex_nli_fallback.csv \
  --source_xlsx "/Users/jackie/Downloads/Yu EMNLP/XHS_SC_BERT/data/test.xlsx" \
  --output_dir outputs/lexicon/error_cue_extraction/up_to_neu \
  --gold_label 0 \
  --pred_label 1 \
  --limit 200 \
  --model gpt-4.1-nano
```

Review these outputs before merging anything into the main lexicon:

- `candidate_cues_UPWARD_to_NEUTRAL.csv`
- `candidate_cues_summary_UPWARD_to_NEUTRAL.csv`

Mine n-gram candidates from all major error slices without calling an API:

```bash
python scripts/mine_lexicon_gaps_from_results.py \
  --results outputs/frame_lexicon_test/test_framelex_nli_fallback.csv \
  --split_xlsx "/Users/jackie/Downloads/Yu EMNLP/XHS_SC_BERT/data/test.xlsx" \
  --output_dir outputs/lexicon/gap_mining
```

Use the newer LLM gap extractor to propose frame-level rows from UP -> NEUTRAL errors:

```bash
python scripts/llm_extract_lexicon_gap_candidates.py \
  --results outputs/frame_lexicon_test/test_framelex_nli_fallback.csv \
  --split_xlsx "/Users/jackie/Downloads/Yu EMNLP/XHS_SC_BERT/data/test.xlsx" \
  --output outputs/lexicon/gap_mining/llm_up_to_neutral_candidates.csv \
  --gt 0 \
  --pred 1 \
  --limit 200 \
  --model gpt-5
```

NLI retrieval with global fallback:

```bash
python run_xhs_frame_lexicon_global_llm.py \
  --input "/Users/jackie/Downloads/Yu EMNLP/XHS_SC_BERT/data/val.xlsx" \
  --lexicon outputs/lexicon/xhs_social_comparison_lexicon.csv \
  --output outputs/frame_lexicon_val/val_framelex_nli_fallback.csv \
  --base_url https://api.bianxie.ai \
  --model gpt-4.1-nano \
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
  --model gpt-4.1-nano \
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
  --model gpt-4.1-nano \
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
  --model gpt-4.1-nano \
  --num_runs 3 \
  --include_context_rules
```
