# test_train_augmented_stable_nli_neu error analysis

## Overall

Confusion matrix rows=gold, cols=predicted:

```text
pred_label  DOWN  NEUTRAL   UP
gt_label                      
DOWN         497      272  162
NEUTRAL      167      483  276
UP            95      378  453
```

Total misclassified: 1350 / 2783

Error counts:

- UP->NEUTRAL: 378
- NEUTRAL->UP: 276
- DOWN->NEUTRAL: 272
- NEUTRAL->DOWN: 167
- DOWN->UP: 162
- UP->DOWN: 95

## Dominant frames among errors

- DOWN->NEUTRAL: {'NONE': 272}
- DOWN->UP: {'none': 40, 'appearance_body_success': 21, 'NONE': 20, 'resource_accumulation': 9, 'mobility_peak_experience': 9}
- NEUTRAL->DOWN: {'NONE': 124, 'low_agency_constraint': 16, 'none': 8, 'blocked_aspiration_failure': 7, 'negative_affect_social_pain': 6}
- NEUTRAL->UP: {'none': 69, 'appearance_body_success': 44, 'NONE': 32, 'resource_advantage': 17, 'mobility_peak_experience': 11}
- UP->DOWN: {'NONE': 70, 'low_agency_constraint': 6, 'none': 6, 'blocked_aspiration_failure': 4, 'negative_affect_social_pain': 2}
- UP->NEUTRAL: {'NONE': 378}

## Notes

- Adding NEUTRAL frames reduced UP/DOWN -> NEUTRAL errors but increased NEUTRAL -> UP errors.
- Many wrong rows still report `comparison_relation=absent`, so final labels are not always consistent with the diagnostic fields.
- NLI outputs include non-canonical frame names such as information_sharing/information_request/social_media_tag, suggesting frame-name constraints need tightening.