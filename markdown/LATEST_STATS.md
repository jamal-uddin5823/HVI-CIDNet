Links:
Enhancer: https://arxiv.org/abs/2502.20272
Recogniser: https://arxiv.org/abs/2204.00964
Closest work: https://ieeexplore.ieee.org/document/10476748/
PPT: https://docs.google.com/presentation/d/1UEDbz3J-SuVoD_jM3WBGZqDnSf-P0c-BS5D1Lrhzlsw/edit?usp=sharing
Codebase: https://github.com/jamal-uddin5823/HVI-CIDNet

Stats:
Metrics measured, Category / Metric,   Meaning, No Enhancement,  Baseline Enhancement,    fr_0.5 Enhancement,  Interpretation
Genuine Similarity (Low-light),  Similarity between images of the same person before enhancement (higher = better),   0.5773 ± 0.2425, 0.6239 ± 0.1993, 0.6078 ± 0.2030, Baseline gives best low-light matchability.
Genuine Similarity (Enhanced),   Similarity between same person after enhancement,    0.5773 (unchanged),  0.9973 ± 0.0021, 0.9979 ± 0.0017, Both near-perfect; fr_0.5 slightly higher.
Impostor Similarity (Low-light), Similarity between different people before enhancement (lower = better), 0.5174 ± 0.2424, 0.5742 ± 0.1814, 0.5535 ± 0.1934, No-enhancement has best impostor separation in raw form.
Impostor Similarity (Enhanced),  Similarity between different people after enhancement (lower = better),  0.5174,  0.4602 ± 0.1236, 0.4639 ± 0.1251, Both enhanced versions improve separation; baseline slightly better.
Similarity Improvement,  Gain in genuine similarity from enhancement, 0.0000,  0.3734,  0.3901,  fr_0.5 yields the strongest identity recovery.
EER (Low-light), Error rate before enhancement (lower = better),  46.81%,  42.55%,  43.62%,  Baseline best pre-enhancement.
EER (Enhanced),  Error rate after enhancement,    46.81%,  0.00%,   0.00%,   Both enhancements lead to perfect verification.
TAR @ FAR 0.1% / 1% (Low-light), True accept rate under strict conditions (higher = better),  2.13%,   0–2.13% 2.13%,   Performance nearly random without enhancement.
TAR @ FAR 0.1% / 1% (Enhanced),  Acceptance rate after enhancement,   2.13%,   100%,    100%,    Both enhancement methods achieve perfect verification.
PSNR,    Measures visual fidelity (higher = sharper / less noise),    7.09 dB, 35.43 dB,    35.29 dB,    Enhancement dramatically improves image clarity.
SSIM,    Perceptual similarity (higher = more natural structure preserved),   0.0753,  0.9559,  0.9551,  Both enhanced images preserve structure well; baseline slightly better.
