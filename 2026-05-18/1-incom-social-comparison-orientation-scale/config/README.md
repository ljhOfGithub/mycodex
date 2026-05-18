# API key management

Use a local env file for API credentials instead of putting keys in scripts or shell history.

1. Copy `config/api_keys.example.env` to `config/api_keys.local.env`.
2. Fill in `NEWAPI_API_KEY`.
3. Run the evaluation script normally. It loads `config/api_keys.local.env` by default.

You can override the default file with:

```bash
python run_xhs_frame_lexicon_global_llm.py \
  --env_file config/another_keys.secret.env \
  --input "/Users/jackie/Downloads/Yu EMNLP/XHS_SC_BERT/data/val.xlsx" \
  --output outputs/frame_lexicon_val/val_framelex_top_all.csv
```

Precedence order:

1. `--api_key`
2. environment variables `NEWAPI_API_KEY` / `OPENAI_API_KEY`
3. values in `--env_file`

`config/api_keys.local.env` and `config/*.secret.env` are ignored by git.
