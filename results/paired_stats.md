Paired comparisons vs raw (n_test: SST-2 672, BoolQ 600); 38 tests, Holm-corrected

| task | ckpt | framing | method | acc raw | acc method | diff [95% CI] | raw-only / method-only correct | McNemar p | Holm p | sig | dECE [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SST-2 | english | noul 'positive?' | contextual | 0.488 | 0.448 | -0.040 [-0.112, +0.027] | 294 / 267 | 2.72e-01 | 1.00e+00 |  | -0.337 [-0.401, -0.265] |
| SST-2 | english | noul 'positive?' | platt | 0.488 | 0.625 | +0.137 [+0.082, +0.192] | 154 / 246 | 4.90e-06 | 1.22e-04 | + | -0.444 [-0.485, -0.394] |
| SST-2 | english | noul 'negative?' | contextual | 0.603 | 0.762 | +0.159 [+0.101, +0.217] | 147 / 254 | 1.02e-07 | 2.74e-06 | + | -0.128 [-0.176, -0.072] |
| SST-2 | english | noul 'negative?' | platt | 0.603 | 0.835 | +0.232 [+0.182, +0.283] | 86 / 242 | 2.55e-18 | 8.93e-17 | + | -0.250 [-0.282, -0.203] |
| SST-2 | english | choice yes=A | contextual | 0.777 | 0.869 | +0.092 [+0.062, +0.122] | 23 / 85 | 1.52e-09 | 4.56e-08 | + | -0.085 [-0.112, -0.042] |
| SST-2 | english | choice yes=A | platt | 0.777 | 0.868 | +0.091 [+0.061, +0.121] | 22 / 83 | 1.62e-09 | 4.69e-08 | + | -0.082 [-0.106, -0.051] |
| SST-2 | english | choice yes=B | contextual | 0.735 | 0.832 | +0.097 [+0.054, +0.140] | 84 / 149 | 2.46e-05 | 5.91e-04 | + | -0.095 [-0.126, -0.054] |
| SST-2 | english | choice yes=B | platt | 0.735 | 0.862 | +0.126 [+0.095, +0.159] | 26 / 111 | 1.05e-13 | 3.57e-12 | + | -0.114 [-0.137, -0.080] |
| SST-2 | english | choice averaged | contextual | 0.754 | 0.874 | +0.119 [+0.082, +0.156] | 43 / 123 | 3.98e-10 | 1.23e-08 | + | -0.066 [-0.092, -0.024] |
| SST-2 | english | choice averaged | platt | 0.754 | 0.876 | +0.122 [+0.091, +0.155] | 26 / 108 | 4.48e-13 | 1.48e-11 | + | -0.090 [-0.110, -0.054] |
| BoolQ | english | noul | contextual | 0.780 | 0.728 | -0.052 [-0.085, -0.020] | 66 / 35 | 2.65e-03 | 5.57e-02 |  | +0.082 [+0.054, +0.109] |
| BoolQ | english | noul | contextual_per_item | 0.780 | 0.675 | -0.105 [-0.142, -0.068] | 98 / 35 | 4.29e-08 | 1.20e-06 | - | +0.145 [+0.104, +0.174] |
| BoolQ | english | noul | platt | 0.780 | 0.780 | +0.000 [+0.000, +0.000] | 0 / 0 | 1.00e+00 | 1.00e+00 |  | +0.007 [-0.034, +0.049] |
| BoolQ | english | choice yes=A | contextual | 0.768 | 0.770 | +0.002 [-0.003, +0.007] | 1 / 2 | 1.00e+00 | 1.00e+00 |  | +0.005 [-0.012, +0.016] |
| BoolQ | english | choice yes=A | contextual_per_item | 0.768 | 0.727 | -0.042 [-0.067, -0.017] | 44 / 19 | 2.23e-03 | 4.90e-02 | - | +0.040 [+0.007, +0.058] |
| BoolQ | english | choice yes=A | platt | 0.768 | 0.790 | +0.022 [-0.003, +0.047] | 23 / 36 | 1.17e-01 | 1.00e+00 |  | -0.031 [-0.062, +0.015] |
| BoolQ | english | choice yes=B | contextual | 0.783 | 0.743 | -0.040 [-0.070, -0.012] | 50 / 26 | 7.91e-03 | 1.50e-01 |  | +0.049 [+0.011, +0.073] |
| BoolQ | english | choice yes=B | contextual_per_item | 0.783 | 0.693 | -0.090 [-0.127, -0.055] | 88 / 34 | 1.08e-06 | 2.81e-05 | - | +0.084 [+0.042, +0.114] |
| BoolQ | english | choice yes=B | platt | 0.783 | 0.803 | +0.020 [+0.005, +0.035] | 5 / 17 | 1.69e-02 | 3.04e-01 |  | -0.027 [-0.055, +0.018] |
| SST-2 | multilingual | noul 'positive?' | contextual | 0.494 | 0.592 | +0.098 [+0.024, +0.167] | 259 / 325 | 7.10e-03 | 1.42e-01 |  | -0.267 [-0.329, -0.200] |
| SST-2 | multilingual | noul 'positive?' | platt | 0.494 | 0.756 | +0.262 [+0.210, +0.310] | 72 / 248 | 8.74e-24 | 3.32e-22 | + | -0.447 [-0.475, -0.386] |
| SST-2 | multilingual | noul 'negative?' | contextual | 0.552 | 0.802 | +0.250 [+0.202, +0.302] | 71 / 239 | 2.04e-22 | 7.55e-21 | + | -0.346 [-0.386, -0.297] |
| SST-2 | multilingual | noul 'negative?' | platt | 0.552 | 0.795 | +0.243 [+0.196, +0.292] | 68 / 231 | 6.36e-22 | 2.29e-20 | + | -0.357 [-0.396, -0.308] |
| SST-2 | multilingual | choice yes=A | contextual | 0.707 | 0.677 | -0.030 [-0.088, +0.025] | 196 / 176 | 3.25e-01 | 1.00e+00 |  | -0.027 [-0.080, +0.025] |
| SST-2 | multilingual | choice yes=A | platt | 0.707 | 0.833 | +0.126 [+0.088, +0.162] | 43 / 128 | 5.30e-11 | 1.70e-09 | + | -0.167 [-0.192, -0.122] |
| SST-2 | multilingual | choice yes=B | contextual | 0.814 | 0.823 | +0.009 [-0.024, +0.042] | 64 / 70 | 6.66e-01 | 1.00e+00 |  | +0.008 [-0.022, +0.037] |
| SST-2 | multilingual | choice yes=B | platt | 0.814 | 0.836 | +0.022 [-0.004, +0.049] | 35 / 50 | 1.28e-01 | 1.00e+00 |  | -0.031 [-0.050, -0.004] |
| SST-2 | multilingual | choice averaged | contextual | 0.772 | 0.811 | +0.039 [-0.003, +0.080] | 90 / 116 | 8.13e-02 | 1.00e+00 |  | -0.019 [-0.056, +0.017] |
| SST-2 | multilingual | choice averaged | platt | 0.772 | 0.836 | +0.064 [+0.030, +0.098] | 53 / 96 | 5.36e-04 | 1.23e-02 | + | -0.039 [-0.074, -0.012] |
| BoolQ | multilingual | noul | contextual | 0.655 | 0.643 | -0.012 [-0.048, +0.025] | 64 / 57 | 5.86e-01 | 1.00e+00 |  | +0.064 [+0.033, +0.104] |
| BoolQ | multilingual | noul | contextual_per_item | 0.655 | 0.610 | -0.045 [-0.085, -0.005] | 84 / 57 | 2.82e-02 | 4.79e-01 |  | +0.105 [+0.067, +0.144] |
| BoolQ | multilingual | noul | platt | 0.655 | 0.658 | +0.003 [-0.013, +0.022] | 14 / 16 | 8.56e-01 | 1.00e+00 |  | -0.166 [-0.194, -0.107] |
| BoolQ | multilingual | choice yes=A | contextual | 0.637 | 0.627 | -0.010 [-0.020, +0.000] | 8 / 2 | 1.09e-01 | 1.00e+00 |  | +0.010 [+0.001, +0.023] |
| BoolQ | multilingual | choice yes=A | contextual_per_item | 0.637 | 0.620 | -0.017 [-0.035, +0.003] | 23 / 13 | 1.32e-01 | 1.00e+00 |  | +0.010 [-0.008, +0.033] |
| BoolQ | multilingual | choice yes=A | platt | 0.637 | 0.618 | -0.018 [-0.035, -0.003] | 17 / 6 | 3.47e-02 | 5.55e-01 |  | -0.245 [-0.273, -0.187] |
| BoolQ | multilingual | choice yes=B | contextual | 0.660 | 0.642 | -0.018 [-0.053, +0.017] | 62 / 51 | 3.47e-01 | 1.00e+00 |  | +0.110 [+0.071, +0.135] |
| BoolQ | multilingual | choice yes=B | contextual_per_item | 0.660 | 0.627 | -0.033 [-0.068, +0.003] | 68 / 48 | 7.73e-02 | 1.00e+00 |  | +0.126 [+0.088, +0.155] |
| BoolQ | multilingual | choice yes=B | platt | 0.660 | 0.655 | -0.005 [-0.023, +0.013] | 20 / 17 | 7.43e-01 | 1.00e+00 |  | -0.059 [-0.094, -0.006] |

Key schemes vs semantic keys (12 tests, Holm-corrected)

| dataset | ckpt | scheme | acc diff | semantic-only / scheme-only correct | McNemar p | Holm p |
|---|---|---|---|---|---|---|
| AG News | english | letters | +0.005 | 2 / 3 | 1.000 | 1.000 |
| AG News | english | numbers | +0.000 | 3 / 3 | 1.000 | 1.000 |
| AG News | english | random | +0.005 | 2 / 3 | 1.000 | 1.000 |
| Emotion | english | letters | -0.017 | 17 / 12 | 0.458 | 1.000 |
| Emotion | english | numbers | -0.007 | 13 / 11 | 0.839 | 1.000 |
| Emotion | english | random | +0.013 | 9 / 13 | 0.523 | 1.000 |
| AG News | multilingual | letters | -0.010 | 3 / 1 | 0.625 | 1.000 |
| AG News | multilingual | numbers | -0.015 | 4 / 1 | 0.375 | 1.000 |
| AG News | multilingual | random | -0.005 | 3 / 2 | 1.000 | 1.000 |
| Emotion | multilingual | letters | +0.063 | 12 / 31 | 0.005 | 0.065 |
| Emotion | multilingual | numbers | +0.060 | 16 / 34 | 0.015 | 0.169 |
| Emotion | multilingual | random | +0.010 | 18 / 21 | 0.749 | 1.000 |
