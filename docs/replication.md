# Replication notes

## Models

The paper reports Python 3.12.4 and spaCy 3.6.0. The scripts specify NumPy 1.26.4 and contain installation comments for these language models:

- German: `de_core_news_lg` 3.6.0.
- Spanish: `es_core_news_lg` 3.8.0.
- English: `en_core_web_lg` 3.7.1.

These model releases require different spaCy minor versions and cannot all be installed with compatible requirements in one environment. The original scripts process all three languages in one process. The author confirmed that these versions were used. Preserve these releases in separate compatible spaCy environments. The supplied all-language scripts still require language-section execution in those environments; an automated split or runner has not been introduced. `requirements.txt` records imported packages, not a validated or historically exact environment. `sentence-transformers` was mentioned in installation comments but is not imported.

fastText uses `cc.de.300.bin`, `cc.es.300.bin`, `cc.en.300.bin`. The scripts download missing models into the working directory. Obtain the original binaries or allow downloads from the fastText distribution (`dl.fbaipublicfiles.com`); allow several GB per language plus memory for model loading.

BERT models are `dbmdz/bert-base-german-cased`, `dccuchile/bert-base-spanish-wwm-cased`, and `bert-base-cased`. Transformers downloads them through Hugging Face unless cached. Original model revisions are not pinned. An exact numerical replication needs the original model artifacts and dependency versions. Times New Roman is requested by plotting code; a missing font changes figure appearance.

## Results alignment and remaining limitations

- Figure 3 panels A, C and E show ID regressions without token count; panels B, D and F show PCA regressions with token count. Retained the existing raw `id_bert`, `x=None` calls and PCA calls. Removed the alternative matrix-ID regression calls that overwrote these files, including English calls using Spanish output names. No formula or estimator implementation was changed. This selects the supplied calls consistent with the published panel covariates; numerical agreement remains unverified without the private data.
- The paper's methods describe token-count adjustment for BERT generally, whereas its Figure 3 ID panels omit it. The figure-specific calls are retained without adding covariates. Existing GEE clustering is unchanged.
- Enabled the existing `sensitivity_all_{name}.jpg` save command with author approval. Figure S4 now has its component-image save step before image assembly; Figure S5 follows it.
- Supplied sensitivity-vector column names use list positions. The code has been retained without reinterpreting positions as actual k values; confirm the intended indexing before interpreting the plotted neighbourhood labels.
- The supplied code does not assemble Figure S1 from the three word-count plots, and does not explicitly assemble Figures S2–S3. It also does not provide a complete post-hoc all-content-word replication pipeline. These outputs require additional instructions/code from the authors.
- PCA scaling, BERT overlapping chunks, model mode, ID on raw/correlation/similarity matrices, missing-value handling and model fitting are unchanged. No seed or new filtering rule was introduced.

## Removed material

Original uploaded filenames and analysis order are preserved. Windows user paths are replaced with `/PATH/TO/PRIVATE_DATA/.../` and `/PATH/TO/RESULTS/` literals; no configuration module or runner was introduced.

- `0_demographics.py`: removed the inactive triple-quoted Spanish statistics example. The active statistics are retained in `2_figures_regressions.py`.
- `1_dim_reduction.py`: removed uncalled mutual-information and eigenspectrum plotting functions. Removed Spanish post-save subsets/averaging and the English post-save averages, group/picture subsets and Shapiro exploration. These do not generate the tables read downstream; the original Shapiro tail referenced undefined German subsets. All per-sample calculations and saved tables, including auxiliary spectrum/centroid/projection outputs, are retained to avoid changing the extraction logic.
- `2_figures_regressions.py`: removed alternative correlation/similarity-matrix ID regression calls from the results section so they no longer overwrite the selected raw-ID panels.
- `3_ID_bert_layers.py`: retained extraction through the three saved language tables. Removed subsequent exploratory plots/regressions; these reference undefined `df_ge` and are not needed by the Figure 4 script.
- `9_sensitivity_figures.py`: retained processing through Figure S5. Removed the subsequent additional ID–similarity plotting exploration; Figure 5 is produced by scripts 5 and 7.
- Removed imports with no remaining references in retained scripts. Existing installation comments are left as historical guidance, subject to the model compatibility issue above.

The following complete attachments are omitted because they are exploratory extensions rather than analyses reported in the supplied paper: `8_pca_projection.py` (animations), `9_summary_tables_recurrences.py` (recurrence/symptom exploration), `10_autoencoders.py`, `11_dim_reduction_synt.py`. Original uploads remain available outside the checkout. The article PDF is referenced by DOI rather than copied into the repository.

## Validation

All retained scripts parse as Python. Every retained top-level statement, after normalizing path values, matches an original statement in the original order. No calculation was rewritten. The sensitivity plotting function has one approved change: its existing image save command is enabled. End-to-end execution, dependency resolution, data schema checks and numerical agreement require private inputs, original model artifacts and the execution arrangements and checks described above; they have not been validated.
