# lexical-meaning-SciRep2025

Code for *Lexical meaning is lower dimensional in psychosis*, Scientific Reports 16, 859 (2026). https://doi.org/10.1038/s41598-025-30443-1

## Setup

The paper reports Python 3.12.4 and NumPy 1.26.4 is specified in the supplied scripts. Dependencies are listed in `requirements.txt`; the original environment lockfile was not supplied. Language-specific model versions listed in [replication notes](docs/replication.md); they require separate compatible spaCy environments.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Data and paths

Datasets are not included and are not publicly available without access restrictions. Obtain the original data through the corresponding author. No example observations are supplied.

Replace the `# PATH INPUT` / `# PATHS` values in each script with absolute paths ending in `/`:

| Variable | Placeholder directory |
| --- | --- |
| `path_input` | `data/german/texts/` |
| `path_demo` | `data/german/` |
| `path_santander` | `data/spanish/` |
| `path_london` | `data/english/` |
| `path_results` | `results/` |

These directories are empty placeholders. Required filenames and columns are listed in [data/README.md](data/README.md). Keep the original identifiers, group labels and picture codes. Demographic preparation writes derived files into the Spanish and English input directories.

## Order

Run from `models/` so the fastText downloads are kept outside the source directories. Use an absolute script path or `../scripts/<script>` from that directory. Scripts run at import time and show figures interactively.

| Order | Script in `scripts/` | Result |
| --- | --- | --- |
| 1 | `0_demographics.py` | Spanish and English demographics |
| 2 | `1_dim_reduction.py` | PCA, ID, similarity and dispersion; three language result tables |
| 3 | `2_figures_regressions.py` | Group regressions; Figures 1–3 |
| 4 | `3_ID_bert_layers.py` | Layer-wise BERT ID tables |
| 5 | `4_figures_bert_layers.py` | Figure 4 |
| 6 | `5_corr_semsim_reducibility.py` | Correlation heatmaps and regressions |
| 7 | `6_count_words.py` | Word-count comparisons; Figure S1 component plots |
| 8 | `7_figures_correlations.py` | Figure 5, using step 6 heatmaps |
| 9 | `8_sensitivity.py` | ID estimates over neighbourhood sizes |
| 10 | `9_sensitivity_figures.py` | Figures S4–S5 |

Steps 4 and 9 can run after the private inputs are configured; plotting scripts require their preceding result tables. The numbers in the original filenames are preserved.

## Replication status

The retained code preserves the supplied calculations, model names and statistical specifications. Original paths were replaced and unused imports/functions and exploratory sections were removed.
