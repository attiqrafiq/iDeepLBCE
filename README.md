# iDeepLBCE

**A deep learning framework and online server for predicting linear B-cell epitopes and non-epitopes.**

iDeepLBCE uses a convolutional neural network (CNN) to classify peptide sequences for B-cell epitope prediction. This repository provides the study datasets, precomputed features, and notebooks for training, cross-validation, and independent evaluation, including ESM-2 fine-tuning, frozen ESM-2 embedding baselines, experiments on a Beetle-derived dataset, and a source-antigen overlap audit. Dataset sources and experimental details are described in the accompanying manuscript.

## Online server

Open the [iDeepLBCE online server](https://ideeplbce.streamlit.app/) to:

- **Predict:** submit sequences and view or download prediction results.
- **Encode:** extract PSTPP features and download them as CSV files.
- **Dataset:** download the benchmark datasets.

See the [illustrated online server guide](README_online_server.md) for step-by-step instructions and screenshots.

## Resources in this repository

| Resource | Location | Purpose |
|---|---|---|
| Main FASTA datasets | [dataset/](dataset/) | iBCE training and internal test sets, external CLBE set, and external ABCPred set |
| Precomputed features | [features/](features/) | Dataset-specific feature CSVs for the main experiments |
| Training and independent evaluation | [ideeplbce-heldout-train-evaluation.ipynb](ideeplbce-heldout-train-evaluation.ipynb) | Train selected models and evaluate on internal and external sets |
| Cross-validation | [ideeplbce-kfold-cross-validation.ipynb](ideeplbce-kfold-cross-validation.ipynb) | Perform stratified k-fold cross-validation using the training set |
| ESM-2 fine-tuning | [esm2-fine-tune-evaluate.ipynb](ESM2_fine_tune_notebook/esm2-fine-tune-evaluate.ipynb) | Fine-tune and evaluate ESM-2 directly from FASTA sequences |
| Frozen ESM-2 baselines | [frozen-esm2-35m-150m.ipynb](ESM2_fine_tune_notebook/frozen-esm2-35m-150m.ipynb) | Extract frozen 35M/150M ESM-2 embeddings and evaluate training-fitted Logistic Regression classifiers on test sets |
| Source-antigen overlap audit | [antigen_overlap_audit/](antigen_overlap_audit/) | Audit training/test source-antigen overlap using IEDB metadata; includes the notebook and generated results archive |
| Beetle-derived dataset experiments | [Performance_on_Beetle_Dataset/](Performance_on_Beetle_Dataset/) | Dataset preparation materials, FASTA splits, PSTPP features, and evaluation notebook |
| Notebook dependencies | [environment.yml](environment.yml) and [requirements.txt](requirements.txt) | Environment specifications for running the research notebooks |

## Documentation

- [Online server guide](README_online_server.md): illustrated instructions for prediction, PSTPP feature extraction, and dataset downloads.
- [Notebook guide](README_overall_notebooks.md): project structure, dataset and feature descriptions, notebook workflows, environment setup, and saved outputs.

The notebooks can be run locally, on Kaggle, or in Google Colab after configuring the required libraries and file paths. Each notebook includes execution guidance. For local setup, follow the environment instructions in the [notebook guide](README_overall_notebooks.md#environment-setup).
