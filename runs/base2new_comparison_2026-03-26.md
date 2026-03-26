# Base-to-New Comparison (2026-03-26)

## Setting

- Protocol matched to the original CoCoOp base/new benchmark: `ViT-B/16`, `16-shot`, `10 epochs`, `base/new split`, seeds `1,2,3`.
- Datasets completed: `EuroSAT`, `FGVC-Aircraft`, `DTD`, `Food-101`, `UCF101`.
- Metrics are `Base`, `New`, and harmonic mean `H`.
- `CoCoOp reproduced here` is available only for `EuroSAT` and `FGVC-Aircraft`; the newly added three datasets were not rerun for CoCoOp in this phase per the experimental plan.

## Important Caveats

- The repository did not include the required caption JSON files for `LiCoCoOp` and `DuCoCoOp`.
- Captions were regenerated locally from the project appendix prompt templates and lightly cleaned where generation produced malformed text.
- `DuCoCoOp` uses the fixed gated-fusion path, frozen BERT/caption generator parameters, and build-time caption embedding initialization.
- Dataset loaders were patched so captions follow the active `base/new` class subset instead of the full class list.
- For `DTD/LiCoCoOp`, one train log missed the final base-accuracy block; the base number for `seed2` was recovered with a separate `eval-only` run on the base split.

## Mean Results

| Dataset | Method | Base | New | H |
|---|---|---:|---:|---:|
| EuroSAT | CoCoOp paper (CVPR 2022) | 87.49 | 60.04 | 71.21 |
| EuroSAT | CoCoOp reproduced here | 88.00 | 62.93 | 73.39 |
| EuroSAT | LiCoCoOp | 88.20 | 70.50 | 78.36 |
| EuroSAT | DuCoCoOp | 86.87 | 60.67 | 71.44 |
| FGVC-Aircraft | CoCoOp paper (CVPR 2022) | 33.41 | 23.71 | 27.74 |
| FGVC-Aircraft | CoCoOp reproduced here | 35.33 | 32.77 | 34.00 |
| FGVC-Aircraft | LiCoCoOp | 32.73 | 23.57 | 27.40 |
| FGVC-Aircraft | DuCoCoOp | 35.17 | 32.50 | 33.78 |
| DTD | CoCoOp paper (CVPR 2022) | 77.01 | 56.00 | 64.85 |
| DTD | LiCoCoOp | 76.07 | 54.57 | 63.55 |
| DTD | DuCoCoOp | 76.63 | 54.70 | 63.84 |
| Food-101 | CoCoOp paper (CVPR 2022) | 90.70 | 91.29 | 90.99 |
| Food-101 | LiCoCoOp | 90.63 | 91.67 | 91.15 |
| Food-101 | DuCoCoOp | 90.40 | 91.13 | 90.77 |
| UCF101 | CoCoOp paper (CVPR 2022) | 82.33 | 73.45 | 77.64 |
| UCF101 | LiCoCoOp | 83.20 | 75.07 | 78.92 |
| UCF101 | DuCoCoOp | 83.03 | 77.13 | 79.97 |

## Seed Results

### EuroSAT

| Method | Seed | Base | New | H |
|---|---:|---:|---:|---:|
| CoCoOp | 1 | 86.7 | 44.0 | 58.37 |
| CoCoOp | 2 | 88.5 | 75.4 | 81.43 |
| CoCoOp | 3 | 88.8 | 69.4 | 77.91 |
| LiCoCoOp | 1 | 84.2 | 64.3 | 72.92 |
| LiCoCoOp | 2 | 88.5 | 73.0 | 80.01 |
| LiCoCoOp | 3 | 91.9 | 74.2 | 82.11 |
| DuCoCoOp | 1 | 87.7 | 65.1 | 74.73 |
| DuCoCoOp | 2 | 82.6 | 54.4 | 65.60 |
| DuCoCoOp | 3 | 90.3 | 62.5 | 73.87 |

### FGVC-Aircraft

| Method | Seed | Base | New | H |
|---|---:|---:|---:|---:|
| CoCoOp | 1 | 34.6 | 33.5 | 34.04 |
| CoCoOp | 2 | 35.6 | 32.2 | 33.81 |
| CoCoOp | 3 | 35.8 | 32.6 | 34.13 |
| LiCoCoOp | 1 | 33.3 | 31.7 | 32.48 |
| LiCoCoOp | 2 | 35.1 | 32.8 | 33.91 |
| LiCoCoOp | 3 | 29.8 | 6.2 | 10.26 |
| DuCoCoOp | 1 | 36.1 | 32.0 | 33.93 |
| DuCoCoOp | 2 | 33.4 | 32.9 | 33.15 |
| DuCoCoOp | 3 | 36.0 | 32.6 | 34.22 |

### DTD

| Method | Seed | Base | New | H |
|---|---:|---:|---:|---:|
| LiCoCoOp | 1 | 76.7 | 53.0 | 62.68 |
| LiCoCoOp | 2 | 74.8 | 53.3 | 62.25 |
| LiCoCoOp | 3 | 76.7 | 57.4 | 65.66 |
| DuCoCoOp | 1 | 75.7 | 57.2 | 65.16 |
| DuCoCoOp | 2 | 78.4 | 52.7 | 63.03 |
| DuCoCoOp | 3 | 75.8 | 54.2 | 63.21 |

### Food-101

| Method | Seed | Base | New | H |
|---|---:|---:|---:|---:|
| LiCoCoOp | 1 | 90.9 | 91.8 | 91.35 |
| LiCoCoOp | 2 | 90.7 | 91.3 | 91.00 |
| LiCoCoOp | 3 | 90.3 | 91.9 | 91.09 |
| DuCoCoOp | 1 | 90.5 | 90.6 | 90.55 |
| DuCoCoOp | 2 | 90.4 | 91.1 | 90.75 |
| DuCoCoOp | 3 | 90.3 | 91.7 | 90.99 |

### UCF101

| Method | Seed | Base | New | H |
|---|---:|---:|---:|---:|
| LiCoCoOp | 1 | 83.2 | 72.5 | 77.48 |
| LiCoCoOp | 2 | 83.4 | 75.8 | 79.42 |
| LiCoCoOp | 3 | 83.0 | 76.9 | 79.83 |
| DuCoCoOp | 1 | 82.7 | 77.8 | 80.18 |
| DuCoCoOp | 2 | 83.2 | 78.3 | 80.68 |
| DuCoCoOp | 3 | 83.2 | 75.3 | 79.05 |

## Interpretation

- `EuroSAT`: `LiCoCoOp` remains the strongest method among the tested variants and stays clearly above the CoCoOp paper reference. `DuCoCoOp` does not beat the reproduced local CoCoOp baseline.
- `FGVC-Aircraft`: neither `LiCoCoOp` nor `DuCoCoOp` surpasses the reproduced local CoCoOp baseline. `DuCoCoOp` stays close, while `LiCoCoOp` remains unstable.
- `DTD`: both language-aware variants underperform the CoCoOp paper reference, with `DuCoCoOp` only marginally above `LiCoCoOp` in harmonic mean.
- `Food-101`: both variants are strong and close to the CoCoOp paper reference, with `LiCoCoOp` slightly ahead of `DuCoCoOp` and slightly above the CoCoOp paper harmonic mean.
- `UCF101`: both variants exceed the CoCoOp paper reference in harmonic mean, and `DuCoCoOp` is the best of the two variants on this dataset.
- Because the caption JSONs were regenerated rather than author-provided, these numbers should be described as a controlled reproduction with regenerated linguistic inputs, not as an exact reconstruction of the original unpublished Li/Du caption pipeline.

## Source Note

- CoCoOp paper reference values are taken from Table 1 of the official CVPR 2022 paper: https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Conditional_Prompt_Learning_for_Vision-Language_Models_CVPR_2022_paper.pdf
