# Train-only Supervised Lexicon Mining Commands

These commands only use `train.xlsx`; do not use val/test to build frames or cues.

## 1. Mine supervised lexical candidates

```bash
/Users/jackie/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/mine_train_supervised_lexicon.py \
  --train_path "/Users/jackie/Downloads/Yu EMNLP/XHS_SC_BERT/data/train.xlsx" \
  --output_dir outputs/lexicon/train_supervised_mining \
  --min_df 5 \
  --prior_strength 500 \
  --hard_negative_folds 5 \
  --hard_negative_top_vocab 3000 \
  --seed_lexicon outputs/lexicon/xhs_social_comparison_lexicon_train_augmented.csv
```

Outputs:

- `train_supervised_ngram_features.csv`: log-odds, chi-square, mutual information, purity/coverage/ambiguity.
- `train_cv_hard_negatives.csv`: train-internal cross-validation error cases for hard-negative mining.
- `train_negation_scope_patterns.csv`: negation / blocked aspiration / self-other gap patterns.
- `train_seed_expansion_candidates.csv`: train-filtered seed expansion candidates.

## 2. Convert selected supervised features to lexicon-compatible candidates

```bash
/Users/jackie/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/convert_supervised_mining_to_lexicon_candidates.py \
  --features outputs/lexicon/train_supervised_mining/train_supervised_ngram_features.csv \
  --output outputs/lexicon/train_supervised_mining/train_supervised_lexicon_candidates.csv \
  --min_purity 0.62 \
  --min_coverage 8 \
  --min_abs_log_odds_z 1.8 \
  --max_rows_per_label 600
```

For review-heavy runs that keep ambiguous markers:

```bash
/Users/jackie/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/convert_supervised_mining_to_lexicon_candidates.py \
  --features outputs/lexicon/train_supervised_mining/train_supervised_ngram_features.csv \
  --output outputs/lexicon/train_supervised_mining/train_supervised_lexicon_candidates_with_ambiguous.csv \
  --min_purity 0.55 \
  --min_coverage 6 \
  --min_abs_log_odds_z 1.5 \
  --include_ambiguous \
  --max_rows_per_label 800
```

## 3. Debug on a balanced subset

```bash
/Users/jackie/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/mine_train_supervised_lexicon.py \
  --train_path "/Users/jackie/Downloads/Yu EMNLP/XHS_SC_BERT/data/train.xlsx" \
  --output_dir outputs/lexicon/train_supervised_mining_pilot_balanced \
  --balanced_limit_per_class 50 \
  --min_df 2
```
