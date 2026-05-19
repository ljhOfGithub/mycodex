# test_train_augmented_stable_nli error analysis

## Overall

Confusion matrix rows=gold, cols=predicted:

```text
predicted  DOWN  NEUTRAL   UP
gold
DOWN        495      326  110
NEUTRAL     164      561  201
UP           97      430  399
```

Total misclassified: 1328 / 2783

Error counts:

- UP->NEUTRAL: 430
- DOWN->NEUTRAL: 326
- NEUTRAL->UP: 201
- NEUTRAL->DOWN: 164
- DOWN->UP: 110
- UP->DOWN: 97

## Dominant frames among errors

- DOWN->NEUTRAL: {'NONE': 326}
- DOWN->UP: {'appearance_body_success': 17, 'NONE': 14, 'none': 14, 'resource_advantage': 10, 'resource_accumulation': 6}
- NEUTRAL->DOWN: {'NONE': 100, 'none': 24, 'low_agency_constraint': 21, 'negative_affect_social_pain': 5, 'blocked_aspiration_failure': 3}
- NEUTRAL->UP: {'appearance_body_success': 41, 'none': 31, 'NONE': 26, 'resource_advantage': 17, 'mobility_peak_experience': 11}
- UP->DOWN: {'NONE': 56, 'none': 14, 'low_agency_constraint': 8, 'negative_affect_social_pain': 3, 'blocked_aspiration_failure': 3}
- UP->NEUTRAL: {'NONE': 429, '享受生活': 1}

## Notes

- Many errors have `dominant_frame=NONE` and `comparison_relation=absent`, indicating remaining neutralization / failure to recognize implicit reader-poster comparison.
- This run used `top_k_neu=0`, so neutralizer frames selected by NLI were not available in the final classifier prompt.
