# Overall Guide to use the Code provided in Notebooks (.ipynb) format

To improve clarity, reproducibility, and ease of use, the notebook files have been revised with explanatory **Markdown sections** and **inline comments where needed**. The aim was to make the workflow more self-explanatory without altering the underlying analytical purpose of each file. In addition, the previously provided `training_validation_kfold.py` script has been converted into a **commented notebook version with the same cleaned logic** so that the execution flow is easier to follow.

The notebooks should be read and used in the following order.

## 1) `dml-keras-tune-evaluate.ipynb`
This notebook is used for **hyperparameter tuning of the deep learning models**. It explores candidate settings for architectures such as CNN, FCNN, RNN, GRU, and LSTM using Keras Tuner, and identifies the best-performing configuration for each model. The selected hyperparameters are then intended for reuse in subsequent experiments.

**Main purpose:** obtain the best model settings before final training and evaluation.

## 2) `esm2-fine-tune-evaluate.ipynb`
This notebook implements the **fine-tuning and evaluation of the ESM-2.0 transformer model** for BCE classification. It includes FASTA parsing, label inference from headers, model fine-tuning, validation, independent testing, and prediction export on benchmark files.

**Main purpose:** benchmark a pre-trained protein language model against the other deep learning approaches.

## 3) `bcell-classical-ml-baselines.ipynb`
This notebook provides the **classical machine learning baseline experiments**. It operates on precomputed feature CSV files and evaluates conventional models to provide a fair baseline comparison against the proposed deep learning framework.

**Main purpose:** establish baseline ML performance for comparison with the deep models.

## 4) `training_validation_kfold.ipynb`
This notebook performs **k-fold cross-validation experiments** across the selected feature sets and deep learning models. It loads the prepared feature files, trains the models fold by fold, records validation predictions, summarizes mean ± standard deviation across folds, and performs statistical comparison using McNemar’s test.

**Main purpose:** assess model stability and comparative performance under cross-validation.

**Note:** this file was previously available as `training_validation_kfold.py`, and it has now been converted into a **commented notebook version with the same cleaned logic** for clarity and ease of understanding.

## 5) `independent_train_evaluate.ipynb`
This notebook is used for **final training and independent evaluation**. After selecting the desired model and feature set, it trains on the training data and evaluates performance on the independent test data. The current workflow is shown using CNN, while the same structure can be applied to the other defined models as noted in the comments.

**Main purpose:** train the final selected configuration and report independent test performance.

# General Notes for Use

- The notebooks are now structured with **section-wise Markdown explanations** and **inline comments** to make execution easier to understand.
- Required input files, folder structure, and output files are noted within the notebooks wherever relevant (already provided in the repo).