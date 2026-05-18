# Lexicon construction notes for XHS-SCoRE

## Goal

Use a frame-level social-comparison lexicon to help prompted LLM classifiers recover reader-grounded UP/DOWN/NEUTRAL comparison cues that are often textually implicit. This targets the generation-detection dissociation described in the XHS-SCoRE paper: LLMs can generate psychologically potent comparison-triggering posts, but prompt-based detectors often neutralize them or skew directionally.

## CACLP adaptation

CACLP builds prompts from lexical knowledge, removes semantic conflicts, and retrieves context-relevant words. For XHS-SCoRE, the equivalent is:

1. Knowledge-enhanced cue generation
   - INCOM -> comparison relation markers: self-other relation, ability/opinion comparison, similarity/dissimilarity, relative standing.
   - UPACS/DACS -> appearance-specific upward/downward comparison frames.
   - LIWC -> auxiliary semantic domains: positive/negative emotion, achievement, money, work, family, body, social, reward/risk.
   - Wmatrix/USAS -> semantic-domain scaffold: achievement, education, money, travel, appearance, relationship, family conflict, low agency, information/tutorial, product/tool, news/ranking.

2. Contrastive refinement
   - UP frames are not merely positive words. They require poster advantage: achievement, resources, attractiveness/body success, social approval.
   - DOWN frames are not merely negative words. They require poster disadvantage: blocked aspiration, low agency, family oppression, body distress, economic/work hardship.
   - Neutralizers prevent false positives: tutorial, product/tool review, third-party news/ranking, casual recommendation.

3. Context-aware retrieval
   - Select only the frames relevant to the post's domain.
   - Give the LLM both cue examples and contrastive rules.
   - Ask the LLM to judge reader-poster positioning, not author sentiment.

## Output files

- `xhs_social_comparison_lexicon.csv`: full lexicon with label, frame, cue, source basis, rationale, rule, and empirical counts in UC/DC/NC files.
- `xhs_social_comparison_lexicon.json`: same content for programmatic use.
- `xhs_social_comparison_lexicon_for_prompt.md`: compact prompt-ready frame lexicon.
- `empirical_xhs_ngram_candidates.csv`: raw data-driven candidate n-grams from UC/DC/NC. This is intentionally noisy and should be used for manual expansion, not directly in prompts.

## Prompt usage pattern

Add the prompt-ready lexicon after the task definition and before examples:

> Use the following frame lexicon as evidence. Do not classify by keyword matching. First ask whether the post positions the poster as advantaged, disadvantaged, or not positioned relative to a young-adult Xiaohongshu reader. Then apply neutralizers for tutorial/product/news posts.

Suggested output rationale fields:

- `comparison_relation`: explicit / implicit / absent
- `dominant_frame`: one frame name from the lexicon
- `neutralizer_present`: yes / no
- `label`: UP / DOWN / NEUTRAL

## Important caveats

- INCOM and UPACS/DACS are scales, not ready-made Chinese lexicons. The extracted cues here are theory-derived frame translations plus XHS-domain expansion.
- LIWC is used only as a category scaffold. The downloaded LIWC manual is not a licensed LIWC dictionary file and should not be treated as one.
- Counts in the CSV are corpus evidence from `UC_full.xlsx`, `DC_full.xlsx`, and `NC_full.xlsx`; high counts do not override the frame rule.
