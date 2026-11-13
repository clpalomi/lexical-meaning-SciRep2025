# Private inputs

Empty directories are placeholders, not datasets. Use the original preprocessed inputs; raw transcription and annotation preparation are not included in the supplied code.

| Directory | File | Columns / format used by the scripts |
| --- | --- | --- |
| `german/texts/` | `<name>_bild_<picture>.txt` | Nonempty UTF-8 transcripts; picture codes 1, 2, 4, 6. `<name>` must match `groups.xlsx`. Files are read recursively. |
| `german/` | `groups.xlsx` | `name`, `group`, `s_group`, `type`, `age`, `education`, `female`. Groups include `HC`, `SSD`, `MDD`; downstream analyses exclude `MDD`. `s_group` and `type` are dropped. |
| `spanish/` | `new_merged_sentences.xlsx` | `PAR`, `Picture`, `Group`, `utterance`. Identifiers `DS_<number>`; pictures `picture_1`, `picture_2`, `picture_3`; groups `controles`, `pacientes`. |
| `spanish/` | `Data_Santander_Discouse.sav` | `codigoFamiliar`, `EdadEvaluacion`, `years_edu`, `sexo`. Demographic IDs are prefixed with `DS_`; female is coded from `sexo == 2`. Filename spelling is retained. |
| `english/` | `summary.xlsx` | `PAR`, `Picture`, `Utterance`, `Role`, `Meta`, `Time`. Headers are lowercased after filtering `Role == interviewee` and `Meta != Off-task`. |
| `english/` | `groups.xlsx` | `par`, `group`. Groups used by the supplied code: `HC`, `FES`, `CHR`, `CS`; `FES` is displayed as FEP in some figures. |
| `english/` | `demo.xlsx` | `ID`, `AgeScan1`, `Gender`; female is coded from `Gender == 2`. |

Columns are inferred from code, not checked against private files. Preserve original types, sentence order, participant IDs and picture codes. German `female` must already use the analysis coding. The scripts expect sufficient words/tokens for PCA and ID neighbour counts; no new exclusion or imputation rules have been added.

`0_demographics.py` creates `df_demographics_santander.pkl` and `df_demographics_london.pkl` in the respective input directories. Other derived tables and images are written to `path_results`.

Data, model downloads and generated results are ignored by Git.
