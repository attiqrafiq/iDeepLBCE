# Overall Guide to the iDeepLBCE Notebooks

This guide describes the datasets, precomputed features, and notebooks currently provided in this project. The notebooks contain Markdown explanations and execution guidelines and can be used locally, on Kaggle, or in Google Colab after configuring input and output paths. The dataset descriptions, sources, and experimental context are provided in the main manuscript.

## 1. Project structure

The hierarchy below excludes `.vscode`. Feature files are summarized by folder; the Beetle preparation artifacts are included for reproducibility.

```text
iDeepLBCE_DOI/
|-- dataset/
|   |-- training set/
|   |   `-- train_ibce.fasta
|   `-- independent test sets/
|       |-- Int-ibce.fasta
|       |-- extI_clbe.fasta
|       `-- extII_abcpred.fasta
|-- features/
|   |-- train_ibce/
|   |-- int_ibce/
|   |-- extI_clbe/
|   `-- extII_abcpred/
|-- ESM2_fine_tune_notebook/
|   |-- esm2-fine-tune-evaluate.ipynb
|   `-- frozen-esm2-35m-150m.ipynb
|-- antigen_overlap_audit/
|   |-- iedb-antigen-overlap-audit-bcell.ipynb
|   `-- iedb_antigen_audit_results.zip
|-- Performance_on_Beetle_Dataset/
|   |-- dataset_beetle/
|   |   |-- train_beetle.fasta
|   |   |-- valid_beetle.fasta
|   |   `-- test _beetle.fasta
|   |-- features_beetle/
|   |   |-- train_beetle/PSTPP_train_beetle.csv
|   |   |-- val_beetle/PSTPP_Val_beetle.csv
|   |   `-- test_beetle/PSTPP_Test_beetle.csv
|   |-- Beetle_dataset_prep_notebooks_files/
|   |   |-- iedb_bcell_dataset_kaggle.ipynb
|   |   |-- beetle_csv_to_fasta.py
|   |   |-- beetle_complete_datset.zip
|   |   |-- beetle_csv_data/
|   |   |   `-- train, valid, and test files in CSV and FASTA formats
|   |   `-- beetle_fasta_files/
|   |       |-- train_beetle.fasta
|   |       |-- valid_beetle.fasta
|   |       `-- test _beetle.fasta
|   `-- ideeplbce-train-evaluation-beetle.ipynb
|-- ideeplbce-heldout-train-evaluation.ipynb
|-- ideeplbce-kfold-cross-validation.ipynb
|-- environment.yml
|-- requirements.txt
|-- README.md
|-- README_overall_notebooks.md
|-- README_online_server.md
`-- readme_images/
```

**Filename note:** the supplied Beetle test FASTA is named `test _beetle.fasta`, including a space before the underscore. Use this exact name when setting a FASTA path. The validation FASTA uses `valid_beetle`, whereas its feature folder uses `val_beetle`.

## 2. Main datasets and feature files

| FASTA file under `dataset/` | Role | Corresponding feature folder |
|---|---|---|
| `training set/train_ibce.fasta` | Training portion of the iBCE dataset | `features/train_ibce/` |
| `independent test sets/Int-ibce.fasta` | Internal held-out test portion of iBCE | `features/int_ibce/` |
| `independent test sets/extI_clbe.fasta` | External CLBE dataset, containing conformational and linear B-cell epitopes | `features/extI_clbe/` |
| `independent test sets/extII_abcpred.fasta` | External ABCPred dataset containing fixed-length sequences | `features/extII_abcpred/` |

Each main feature folder contains precomputed CSV blocks for **AAC, DPC, PAAC, APAAC, PCP, ONEHOT, PSSM, and PSTPP**, together with a label CSV and a corresponding FASTA copy. The notebook feature selectors use the configured feature prefixes. The CLBE FASTA copy inside its feature folder is named `ext1_clbe.fasta`.

The feature-based notebooks read headerless feature CSVs: the first column contains the binary label (`0` = non-epitope; `1` = epitope), and the remaining columns contain numerical features. They read labels directly from these feature CSVs rather than from the separate label files. Preserve sample order across feature blocks when combining them.

Two combinations are defined in the main notebooks:

- **F1:** AAC, DPC, PAAC, APAAC, and PCP.
- **F2:** ONEHOT, PSSM, and PSTPP.

Individual features or custom combinations can be selected according to the notebook settings. Each requested feature prefix must identify exactly one CSV in its split folder; use `CSV_OVERRIDES` when an explicit file selection is needed.

## 3. Main training and independent evaluation

[`ideeplbce-heldout-train-evaluation.ipynb`](ideeplbce-heldout-train-evaluation.ipynb) trains the selected models using `features/train_ibce/` and evaluates them on the internal iBCE, external CLBE, and external ABCPred sets.

The notebook provides classical classifiers (XGBoost, Random Forest, Logistic Regression, and Extra Trees) and deep learning architectures (CNN, FCNN, RNN, LSTM, and GRU). Select features, combinations, models, and repeated-fit seeds through `ML_JOBS`, `DL_JOBS`, and `SEEDS`. Deep learning uses a validation subset of the training partition for early stopping.

**Main purpose:** train the selected configurations and report performance on the internal and external evaluation sets.

By default, `DATA_ROOT = Path('features')`, relative to the runtime's working directory. Results are displayed inline and saved in timestamped subfolders of `_results_/`. Exports include per-sample predictions, per-run metrics, metric summaries, and deep learning training histories/settings.

## 4. Cross-validation on the training set

[`ideeplbce-kfold-cross-validation.ipynb`](ideeplbce-kfold-cross-validation.ipynb) uses only `features/train_ibce/` for stratified k-fold cross-validation. It follows a similar feature-loading and model-building workflow to the held-out notebook. Internal and external test sets are not loaded.

Select `FEATURES`, `MODELS`, and the number of folds, `K`, in Settings. The selected experiments share outer fold assignments. Deep learning uses an additional validation split within each outer training fold for early stopping.

**Main purpose:** assess performance and variability across training-set folds.

Results are shown inline and saved in timestamped subfolders of `_cv_results_/`. Outputs include fold assignments, fold predictions, per-fold metrics, mean and sample standard deviation summaries, pooled out-of-fold predictions/metrics, and applicable training artifacts. Follow the notebook's methodological notes about precomputed label-derived PSTPP features and fold boundaries when interpreting cross-validation results.

## 5. Evaluation on the Beetle-derived dataset

[`Performance_on_Beetle_Dataset/ideeplbce-train-evaluation-beetle.ipynb`](Performance_on_Beetle_Dataset/ideeplbce-train-evaluation-beetle.ipynb) evaluates the proposed methodology using the separate dataset prepared following the Beetle article's guidance. The manuscript provides the source context, and the notebook describes the dataset instances.

| Split | FASTA under `dataset_beetle/` | PSTPP CSV under `features_beetle/` |
|---|---|---|
| Training | `train_beetle.fasta` | `train_beetle/PSTPP_train_beetle.csv` |
| Validation | `valid_beetle.fasta` | `val_beetle/PSTPP_Val_beetle.csv` |
| Test | `test _beetle.fasta` | `test_beetle/PSTPP_Test_beetle.csv` |

This notebook follows a similar model-training and evaluation structure to the main held-out notebook, using **only the supplied PSTPP features**. Deep learning fits on `train_beetle`, uses `val_beetle` for validation and callbacks, and reports validation and test metrics. Reserve the test set for final evaluation. The configured expected split sizes are 119,503 training samples and 10,000 samples each for validation and testing.

**Main purpose:** evaluate the proposed methodology on the additional Beetle-derived dataset.

The default `DATA_ROOT = Path('./features_beetle')` assumes the working directory is `Performance_on_Beetle_Dataset/`. When running from the project root, set it to `Path('Performance_on_Beetle_Dataset/features_beetle')`. Inline results and exported summaries/predictions are produced, with timestamped output directories under `_results_beetle/` relative to the working directory.

### Dataset preparation and reproducibility

`Performance_on_Beetle_Dataset/Beetle_dataset_prep_notebooks_files/` contains the preparation notebook, CSV-to-FASTA script, archived dataset, and intermediate/exported CSV and FASTA files. These materials document dataset preparation following the Beetle workflow described in the manuscript.

The preparation notebook is configured for Kaggle, downloads IEDB data, and builds CD-HIT. Its default outputs are under `/kaggle/working/iedb_bcell/data`. To run it locally or in Colab, adapt its paths and ensure the build tools described in the notebook are available. The supplied FASTA and feature files allow the evaluation notebook to be used without rerunning dataset preparation.

## 6. ESM-2 fine-tuning and evaluation

[`ESM2_fine_tune_notebook/esm2-fine-tune-evaluate.ipynb`](ESM2_fine_tune_notebook/esm2-fine-tune-evaluate.ipynb) provides the ESM-2 training and evaluation workflow. It reads FASTA sequences directly, infers binary labels from headers, and fine-tunes a Hugging Face ESM-2 sequence-classification model. Precomputed feature CSVs are not required.

**Main purpose:** fine-tune and evaluate a protein language model for B-cell epitope classification, with prediction export and optional embedding extraction.

### Workflow and settings

The notebook parses and cleans sequences, creates a stratified training/validation split, and optionally applies class-weighted cross-entropy. It evaluates and saves checkpoints each epoch, uses early stopping, and restores the checkpoint selected by the configured validation metric. Training and prediction Trainers use `report_to=[]`; no W&B account or Kaggle secret is required by this notebook.

The User Settings cell configures `facebook/esm2_t30_150M_UR50D` with maximum token length 64, a 10% validation split, seed 42, learning rate `2e-5`, weight decay `0.01`, training/evaluation batch sizes of 16, 100 learning-rate warmup steps, up to 10 epochs, and early-stopping patience 3. The best checkpoint is selected using validation F1. Adjust the model and batch sizes to the available GPU memory. Internal and external test sets are reserved for independent evaluation.

### Inputs and execution

1. Create the environment described below, select its Jupyter kernel, and leave the notebook's old installation cell commented out to preserve the selected dependency versions.
2. Before importing Transformers, add the following in the ESM-2 kernel to select its PyTorch workflow when TensorFlow/Keras is installed alongside it:

   ```python
   import os
   os.environ["USE_TF"] = "0"
   ```

   Restart the kernel first if Transformers has already been imported.
3. Update `TRAIN_FASTA`, `PREDICT_FASTA`, external prediction paths, and `config.output_dir`. Replace the example Kaggle paths with actual local, Kaggle, or mounted Colab paths. If running from the project root, use `dataset/training set/train_ibce.fasta` for training and the appropriate FASTA under `dataset/independent test sets/` for evaluation. If running from the notebook's own folder, adjust relative paths accordingly.
4. Check that `infer_label()` matches the supplied FASTA header format. Training requires labeled records; prediction accepts labeled or unlabeled files. Evaluation metrics are printed when all prediction records have labels.
5. Run the cells in order. A GPU is recommended. The example enables FP16 when CUDA is available and disables BF16; for CPU execution, keep both disabled. Initial model/tokenizer download requires Internet access unless cached locally.

### Results and saved files

The User Settings cell sets `output_dir="esm2_clean_output"`. Training produces model checkpoints and a `best_model/` directory containing the selected model and tokenizer. It also saves:

- `validation_like_predictions.csv`: predictions on the validation subset.
- `run_summary.json`: a summary of the run configuration.
- Prediction CSVs at the paths passed to `predict()`, containing sequence information, positive-class probabilities, and binary predictions. The default filename is `predictions.csv` inside the output directory.
- Optional embedding arrays from `extract_embeddings()`, using CLS or mean pooling; the default filenames are `embeddings_cls.npy` or `embeddings_mean.npy`.

The notebook displays evaluation metrics inline and includes separate internal/external prediction examples. Review the paths in each example before running them. Optional embedding extraction can be run after training.

## 7. Frozen ESM-2 embedding baselines

[`ESM2_fine_tune_notebook/frozen-esm2-35m-150m.ipynb`](ESM2_fine_tune_notebook/frozen-esm2-35m-150m.ipynb) compares frozen pretrained ESM-2 **35M** (`facebook/esm2_t12_35M_UR50D`) and **150M** (`facebook/esm2_t30_150M_UR50D`) backbones. It extracts final-layer residue embeddings from the training and selected internal/external test FASTAs, mean-pooling them with padding and special tokens excluded. The backbone weights remain unchanged.

For each backbone, a StandardScaler and **Logistic Regression** pipeline is fitted on training data. Five-fold stratified training-only cross-validation selects the regularization parameter `C` by average precision, followed by evaluation on the test sets. These experiments produce single-run point estimates for each backbone; pooling and classifier choices also differ from the fine-tuning workflow.

Update `TRAIN_FASTA`, `TEST_FASTAS`, and `OUTPUT_DIR` for the intended dataset versions and runtime. The example paths refer to curated Kaggle inputs; verify the selected peptides and header labels before comparing experiments. The notebook uses a maximum token length of 64 and rejects longer inputs unless truncation is explicitly enabled. Two GPUs allow parallel backbone processing; one GPU or CPU runs them sequentially. The default output location is `/kaggle/working/frozen_esm2`. Saved outputs include `frozen_ESM2_all_metrics.csv`, per-model metrics and predictions, cached embeddings, fitted classifiers, training CV results, and reproducibility manifests.

## 8. Source-antigen overlap audit

[`antigen_overlap_audit/iedb-antigen-overlap-audit-bcell.ipynb`](antigen_overlap_audit/iedb-antigen-overlap-audit-bcell.ipynb) checks whether training and selected test FASTAs share source proteins or parent antigens using current IEDB metadata. It resolves records through IEDB IDs where available, sequence-based ID recovery, or exact-sequence API lookup, then reports overlap at exact accession, version-normalized accession, and parent-antigen levels. It does not modify the FASTAs or train a prediction model.

The supplied [iedb_antigen_audit_results.zip](antigen_overlap_audit/iedb_antigen_audit_results.zip) contains `overlap_summary.csv`, record mappings, per-test-record overlap statuses, shared-antigen lists, unresolved/ambiguous mappings, and retrieval manifests, request logs, and cached responses. The archived reports identify possible overlap in the audited inputs, with incomplete external-set mapping. Sequence-based external matches provide candidate provenance; unresolved records or no observed overlap do not establish antigen independence. Consult the notebook's report interpretation notes and archived input manifest when relating these results to particular dataset versions.

The notebook is configured for Kaggle and needs Internet access for IEDB retrieval; CPU is sufficient. Select the training/test FASTAs, run the offline checks and small live API test, then set `RUN_FULL_AUDIT = True` for the full retrieval. Reports and downloadable archives are generated under `/kaggle/working/iedb_antigen_audit` on Kaggle, or `iedb_antigen_audit/` under the working directory elsewhere. Input/output paths can be adapted for local or Colab execution; manual file selection is available when widgets are unavailable.

## 9. Running locally, on Kaggle, or in Colab

1. Choose the notebook for the desired experiment. The cross-validation, independent evaluation, ESM-2 fine-tuning, frozen ESM-2, Beetle, and antigen-audit workflows can be run separately.
2. Make the required feature folders or FASTA files available to the runtime, preserving the split-folder structure.
3. Prepare the required libraries and select the notebook kernel. `environment.yml` provides a Conda environment specification, and `requirements.txt` provides a pip dependency list; review the compatibility notes below before using them.
4. Edit input paths and a writable output location. For the main feature notebooks, `DATA_ROOT` must point to the parent containing `train_ibce`, `int_ibce`, `extI_clbe`, and `extII_abcpred`. On Kaggle, use the mounted input path and save outputs under `/kaggle/working`. In Colab, upload the data or mount Drive and use the corresponding `/content/...` paths.
5. Select features, models, folds/seeds, and training options as described in the notebook. Enable a GPU for accelerated deep learning or ESM-2 training.
6. Run the cells in order, inspect inline results, and retain the exported files from the printed results directory.

### Current configuration notes

- The main held-out notebook's introductory table still uses older external folder names. Its executable `DATA_DIRS` and the supplied folders use `extI_clbe` and `extII_abcpred`; use those names. Its `EXPECTED_ROWS` external keys also retain older names, so review them if external row-count checks are required.
- `environment.yml` now selects Python 3.11.13 and installs the shared `requirements.txt` through pip. TensorFlow 2.18.0 is paired with Keras 3.8.0. ESM-2 uses PyTorch 2.6.0, Transformers 4.51.3, and Accelerate 1.6.0. XGBoost, joblib, and ipykernel are explicitly included. W&B remains in the shared requirements but is not used by the documented ESM-2 notebook. The unused `esm==x.x.x` placeholder was removed: the ESM-2 notebook loads models through Hugging Face Transformers.
- The dependency versions are selected compatibility settings, not a complete lockfile. The shared requirement `torch==2.6.0` leaves hardware-specific wheel selection to the installation platform. For a supported Linux/Windows CUDA 12.4 setup, use the official PyTorch wheel command below. Native Windows TensorFlow 2.18 does not provide CUDA GPU support; use WSL2/Linux or a hosted GPU runtime for TensorFlow GPU execution.
- The ESM-2 notebook uses `eval_strategy`, `processing_class`, and a custom loss accepting additional keyword arguments. The selected Transformers version supports these APIs. The User Settings cell configures `warmup_steps=100` consistently with `PipelineConfig` and the Trainer builder.
- Paths are resolved from the runtime's working directory, which can differ between local Jupyter, Kaggle, and Colab. Output directories are created during execution; the antigen audit additionally includes an archived set of generated reports.
- Held-out summaries across seeds and cross-validation summaries across folds describe different sources of variability. With one seed, the held-out sample standard deviation is undefined and exported blank.

### Environment setup

From the repository root, choose either Conda:

```bash
conda env create -f environment.yml
conda activate epitope-env
```

or pip in a fresh Python 3.11 virtual environment:

```bash
python -m pip install -r requirements.txt
```

For a supported CUDA 12.4 PyTorch installation:

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Install the chosen GPU wheel before the shared requirements when creating a fresh environment. For an existing environment, verify `torch.__version__` and `torch.cuda.is_available()`; the base version requirement alone does not replace an already installed CPU wheel. Follow [PyTorch's version-specific installation instructions](https://pytorch.org/get-started/previous-versions/#v260) for other platforms/builds and [TensorFlow's installation guide](https://www.tensorflow.org/install/pip) for TensorFlow GPU setup.

After installation, run `python -m pip check` and register/select the environment's Jupyter kernel if needed:

```bash
python -m ipykernel install --user --name epitope-env --display-name "Python (epitope-env)"
```

The selected TensorFlow/Keras constraints were checked against [TensorFlow 2.18 package metadata](https://pypi.org/pypi/tensorflow/2.18.0/json); the ESM-2 API choices were checked against [Transformers 4.51.3 Trainer documentation](https://huggingface.co/docs/transformers/v4.51.3/en/main_classes/trainer). A full environment installation and model retraining have not been performed as part of this update.
