# Base-to-New Comparison (2026-03-25)

## Setting

- Protocol matched to the original CoCoOp base/new benchmark: `ViT-B/16`, `16-shot`, `10 epochs`, `base/new split`, seeds `1,2,3`.
- Datasets completed: `EuroSAT`, `FGVC-Aircraft`.
- Metrics are `Base`, `New`, and harmonic mean `H`.
- For the mean table below, `H` is computed from the mean `Base` and mean `New`.

## Important Caveats

- The repository did not include the required caption JSON files for `LiCoCoOp` and `DuCoCoOp`.
- Captions were regenerated locally from the appendix prompt templates using `Qwen/Qwen2.5-3B-Instruct` and saved under the dataset roots.
- `DuCoCoOp` had a code path where the gated fusion output was computed and then overwritten; the results below use the fixed gated path.
- Dataset loaders were patched so captions follow the active `base/new` class subset instead of the full class list.
- `BERT` is frozen in these runs and caption embeddings are prepared during model build, so `eval-only` works correctly.

## Mean Results

### EuroSAT

| Method | Base | New | H |
|---|---:|---:|---:|
| CoCoOp paper (CVPR 2022) | 87.49 | 60.04 | 71.21 |
| CoCoOp reproduced here | 88.00 | 62.93 | 73.39 |
| LiCoCoOp | 88.20 | 70.50 | 78.36 |
| DuCoCoOp | 86.87 | 60.67 | 71.44 |

### FGVC-Aircraft

| Method | Base | New | H |
|---|---:|---:|---:|
| CoCoOp paper (CVPR 2022) | 33.41 | 23.71 | 27.74 |
| CoCoOp reproduced here | 35.33 | 32.77 | 34.00 |
| LiCoCoOp | 32.73 | 23.57 | 27.40 |
| DuCoCoOp | 35.17 | 32.50 | 33.78 |

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

## Interpretation

- Against the original CoCoOp paper numbers, `LiCoCoOp` is clearly better on `EuroSAT`, while `DuCoCoOp` is only marginally better in `H`.
- Against the locally reproduced CoCoOp baseline, only `LiCoCoOp` improves on `EuroSAT`; `DuCoCoOp` does not.
- On `FGVC-Aircraft`, neither `LiCoCoOp` nor `DuCoCoOp` improves over the locally reproduced CoCoOp mean. `DuCoCoOp` is close, `LiCoCoOp` is unstable.
- Because the caption JSONs were regenerated rather than author-provided, these numbers should be described as a controlled reproduction with regenerated linguistic inputs, not as an exact reproduction of the original Li/Du paper claims.
