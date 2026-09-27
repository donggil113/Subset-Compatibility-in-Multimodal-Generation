# Calibration contract audit: `configs/p5_e8_calib_v3.json`

Closed form on the Gaussian probe; D values on the standardised scale (y/2, z/0.5). Reference = direct branch, compared = sequential branch.

| cell | target law (cmp vs ref) | projected joint law (cmp vs ref) | cmp vs truth (target) | ref vs truth (target) | Cov(y,z|x) ref -> cmp | sd ratio cmp | E CRPS excess cmp / ref |
|---|---|---|---|---|---|---|---|
| chain_drop_x (markov) | unchanged (D=0.00000) | unchanged (D=0.00000) | 0.00000 | 0.00000 | 0.384 -> 0.384 | 1.000 | 0.0000 / 0.0000 |
| collapse_both (nonmarkov) | unchanged (D=0.00000) | unchanged (D=0.00000) | 0.46505 | 0.46505 | 0.000 -> 0.000 | 0.000 | 0.2325 / 0.2325 |
| collapse_intermediate (nonmarkov) | changes (D=0.02265) | changes (D=0.11007) | 0.02265 | 0.00000 | 0.540 -> 0.000 | 0.735 | 0.0113 / 0.0000 |
| corr_scale_-1 (nonmarkov) | unchanged (D=0.00000) | changes (D=0.06492) | 0.00000 | 0.00000 | 0.540 -> -0.540 | 1.000 | 0.0000 / 0.0000 |
| exact_sequential (nonmarkov) | unchanged (D=0.00000) | unchanged (D=-0.00000) | 0.00000 | 0.00000 | 0.540 -> 0.540 | 1.000 | 0.0000 / 0.0000 |
| mean_shift_0.1 (nonmarkov) | changes (D=0.00561) | changes (D=0.00327) | 0.00561 | 0.00000 | 0.540 -> 0.540 | 1.000 | 0.0028 / 0.0000 |
| var_ratio_1.25 (nonmarkov) | changes (D=0.00369) | changes (D=0.00228) | 0.00369 | 0.00000 | 0.540 -> 0.540 | 1.118 | 0.0018 / 0.0000 |

## Gate operating characteristics (exact binomial; per-test size 0.05)

| cell | role | R | criteria | P(false fail or false EXCESS at size 0.05) | P(flag at size 0.10) |
|---|---|---|---|---|---|
| exact_sequential|target|nonmarkov | null | 400 | EXCESS iff k>=31; tolerance iff k<=29 | 0.019 | 0.964 |
| exact_sequential|joint|nonmarkov | null | 100 | EXCESS iff k>=11 | 0.011 | 0.417 |
| corr_scale_-1|target|nonmarkov | null | 100 | EXCESS iff k>=11 | 0.011 | 0.417 |
| chain_drop_x|target|markov | null | 100 | EXCESS iff k>=11 | 0.011 | 0.417 |
| mean_shift_0.1|target|nonmarkov | alternative | 50 | POWERED iff k>=6 | P(POWERED) at power 0.05/0.2/0.3: 0.038/0.952/0.999 | |
| var_ratio_1.25|target|nonmarkov | alternative | 50 | POWERED iff k>=6 | P(POWERED) at power 0.05/0.2/0.3: 0.038/0.952/0.999 | |
| collapse_intermediate|target|nonmarkov | alternative | 50 | POWERED iff k>=6 | P(POWERED) at power 0.05/0.2/0.3: 0.038/0.952/0.999 | |
| corr_scale_-1|joint|nonmarkov | alternative | 50 | POWERED iff k>=6 | P(POWERED) at power 0.05/0.2/0.3: 0.038/0.952/0.999 | |

P(all null cells pass | exactly valid test) = 0.948; P(gate pass | valid, each power 0.3) = 0.945.

## p-value resolution vs multiplicity

B = 199: p takes values in multiples of 0.0050 with minimum 0.0050; exact size at alpha = 0.05 for a continuous statistic = 0.050. A Bonferroni family of m tests can reject at all only if m <= 10.

| family size m | per-test threshold | attainable size | can reject |
|---|---|---|---|
| 1 | 0.0500 | 0.0500 | True |
| 2 | 0.0250 | 0.0250 | True |
| 4 | 0.0125 | 0.0100 | True |
| 8 | 0.0063 | 0.0050 | True |
| 10 | 0.0050 | 0.0050 | True |
| 11 | 0.0045 | 0.0000 | False |
| 24 | 0.0021 | 0.0000 | False |
