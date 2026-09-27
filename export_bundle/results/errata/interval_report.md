# Interval erratum report (derived; committed summaries unchanged)

A pair of one-sided 95% Clopper-Pearson bounds is an equal-tailed two-sided 90% interval. Gates used the one-sided bounds exactly as pre-registered; nothing about any gate outcome changes.

| run | cell | k/n | rate | one-sided 95% lower | one-sided 95% upper | two-sided 95% CP |
|---|---|---|---|---|---|---|
| p5_e8_calib_v3 | exact_sequential|target|nonmarkov | 16/400 | 0.0400 | 0.0252 | 0.0601 | [0.0230, 0.0641] |
| p5_e8_calib_v3 | exact_sequential|joint|nonmarkov | 5/100 | 0.0500 | 0.0199 | 0.1023 | [0.0164, 0.1128] |
| p5_e8_calib_v3 | corr_scale_-1|target|nonmarkov | 5/100 | 0.0500 | 0.0199 | 0.1023 | [0.0164, 0.1128] |
| p5_e8_calib_v3 | chain_drop_x|target|markov | 4/100 | 0.0400 | 0.0138 | 0.0892 | [0.0110, 0.0993] |
| p5_e8_calib_v3 | mean_shift_0.1|target|nonmarkov | 45/50 | 0.9000 | 0.8012 | 0.9598 | [0.7819, 0.9667] |
| p5_e8_calib_v3 | var_ratio_1.25|target|nonmarkov | 26/50 | 0.5200 | 0.3954 | 0.6427 | [0.3742, 0.6634] |
| p5_e8_calib_v3 | collapse_intermediate|target|nonmarkov | 50/50 | 1.0000 | 0.9418 | 1.0000 | [0.9289, 1.0000] |
| p5_e8_calib_v3 | corr_scale_-1|joint|nonmarkov | 50/50 | 1.0000 | 0.9418 | 1.0000 | [0.9289, 1.0000] |
| p5_e8_calib_v3 | collapse_both|target|nonmarkov | 0/10 | 0.0000 | 0.0000 | 0.2589 | [0.0000, 0.3085] |
| p5_e8_calib_v3 | collapse_both|joint|nonmarkov | 0/10 | 0.0000 | 0.0000 | 0.2589 | [0.0000, 0.3085] |
| p5_e8_exact_v1 | exact_sequential|target | 9/200 | 0.0450 | 0.0237 | 0.0772 | [0.0208, 0.0837] |
| p5_e8_exact_v1 | corr_scale_-1|target | 4/100 | 0.0400 | 0.0138 | 0.0892 | [0.0110, 0.0993] |
| p5_e8_exact_v1 | exact_sequential|joint | 1/50 | 0.0200 | 0.0010 | 0.0914 | [0.0005, 0.1065] |
| p5_e8_exact_v1 | mean_shift_2.0|target | 42/50 | 0.8400 | 0.7298 | 0.9178 | [0.7089, 0.9283] |
| p5_e8_exact_v1 | var_ratio_4.0|target | 5/50 | 0.1000 | 0.0402 | 0.1988 | [0.0333, 0.2181] |
| p5_e8_exact_v1 | corr_scale_-1|joint | 4/50 | 0.0800 | 0.0278 | 0.1738 | [0.0222, 0.1923] |
| p5_e8_exact_v1 | collapse_intermediate|target | 2/50 | 0.0400 | 0.0072 | 0.1206 | [0.0049, 0.1371] |
