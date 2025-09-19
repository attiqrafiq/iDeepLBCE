import os
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import time, 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, recall_score  # ADD
from statsmodels.stats.contingency_tables import mcnemar  # ADD

from tqdm import tqdm

from Bio import SeqIO

import tensorflow as tf
import joblib
import itertools

physical_devices = tf.config.list_physical_devices("GPU")

try:
    tf.config.set_logical_device_configuration(
        physical_devices[0],
        [
            tf.config.LogicalDeviceConfiguration(memory_limit=4096),
            tf.config.LogicalDeviceConfiguration(memory_limit=4096),
            tf.config.LogicalDeviceConfiguration(memory_limit=4096),
            tf.config.LogicalDeviceConfiguration(memory_limit=4096),
        ],
    )
    print("config 1 done")
    logical_devices = tf.config.list_logical_devices("GPU")
    assert len(logical_devices) == 2
    tf.config.set_logical_device_configuration(
        physical_devices[0],
        [
            tf.config.LogicalDeviceConfiguration(memory_limit=4096),
            tf.config.LogicalDeviceConfiguration(memory_limit=4096),
            tf.config.LogicalDeviceConfiguration(memory_limit=4096),
            tf.config.LogicalDeviceConfiguration(memory_limit=4096),
        ],
    )
    print("config 2 done")
except Exception as e:
    print("Cannot modify logical devices once initialized.", e)
    pass


def adjust_sequences(sequence_list):
    """
    Adjusts protein sequences to the length of the longest sequence by padding with 'X'.

    Parameters:
    - sequences (list of str): List of protein sequences.

    Returns:
    - list of str: List of adjusted protein sequences.
    """
    sequences = [
        seq[1] for seq in sequence_list
    ]  # Extract sequences from the input list

    # Find the maximum sequence length
    max_length = max(len(seq) for seq in sequences)

    adjusted_sequences = []
    for seq in sequences:
        padding_needed = max_length - len(seq)
        prefix_padding = padding_needed // 2
        postfix_padding = padding_needed - prefix_padding
        adjusted_seq = "X" * prefix_padding + seq + "X" * postfix_padding
        adjusted_sequences.append(adjusted_seq)

    return adjusted_sequences


def read_fasta(file):
    """
    Reads a FASTA file and returns a list of sequences and their labels.

    Args:
        file: Path to the FASTA file.

    Returns:
        A tuple containing a list of sequences and a list of labels.
    """
    sequences = []
    labels = []
    for record in SeqIO.parse(file, "fasta"):
        sequences.append(str(record.seq))
        if record.id.lower().__contains__("pos"):
            labels.append(1)
        else:
            labels.append(0)
    return sequences, labels


# ========== MODEL BUILDERS ==========
def build_fcnn(input_shape):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_shape,)),
            tf.keras.layers.Dense(
                352,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(
                352,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC"],
    )
    return model


# from tensorflow.keras.layers import RNN, SimpleRNNCell, GRUCell, LSTMCell
def build_rnn(input_shape):
    # tf.config.set_visible_devices([], "GPU")
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Reshape((1, input_shape), input_shape=(input_shape,)),
            tf.keras.layers.RNN(
                tf.keras.layers.SimpleRNNCell(
                    units=48,
                    activation="tanh",
                    use_bias=True,
                    kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                ),
            ),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(
                320,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC"],
    )
    return model


def build_gru(input_shape):
    # tf.config.set_visible_devices([], "GPU")
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Reshape((1, input_shape), input_shape=(input_shape,)),
            tf.keras.layers.GRU(
                32,
                activation="tanh",
                recurrent_activation="sigmoid",
                use_bias=True,
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
                reset_after=False,  # <- disables CuDNN path
                recurrent_dropout=0.1,  # <- also disables CuDNN
                implementation=2,  # <- standard kernel
                return_sequences=False,
            ),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(
                32,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC"],
    )
    return model


def build_lstm(input_shape):
    # tf.config.set_visible_devices([], "GPU")
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Reshape((1, input_shape), input_shape=(input_shape,)),
            tf.keras.layers.LSTM(
                96,
                activation="tanh",
                recurrent_activation="sigmoid",
                kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                use_bias=True,
                recurrent_dropout=0.1,  # <- disables CuDNN
                implementation=2,  # <- standard kernel
                return_sequences=True,
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(
                96,
                activation="tanh",
                recurrent_activation="sigmoid",
                kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                use_bias=True,
                recurrent_dropout=0.1,  # <- disables CuDNN
                implementation=2,  # <- standard kernel
                return_sequences=False,
            ),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(
                32,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC"],
    )
    return model


def build_cnn(input_shape):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Reshape((input_shape, 1), input_shape=(input_shape,)),
            tf.keras.layers.Conv1D(
                filters=64,
                kernel_size=8,
                activation="relu",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
                # bias_regularizer=tf.keras.regularizers.l2(1e-4),
                # kernel_initializer="random_normal",
            ),
            tf.keras.layers.MaxPool1D(2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Conv1D(
                filters=32,
                kernel_size=8,
                activation="relu",
                padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
                # bias_regularizer=tf.keras.regularizers.l2(1e-4),
                # kernel_initializer="random_normal",
            ),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(
                192,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
                # bias_regularizer=tf.keras.regularizers.l2(1e-4),
                # kernel_initializer="random_normal",
            ),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", "AUC"],
    )
    return model


# Map model names to builder functions
MODEL_BUILDERS = {
    "FCNN": build_fcnn,
    "RNN": build_rnn,
    "GRU": build_gru,
    "LSTM": build_lstm,
    "CNN": build_cnn,
}


def encode_features(features_list, fileName, type="fixed"):
    """
    Combines multiple feature lists into a single DataFrame.

    Args:
        features_list: A list of feature lists (each list is a list of features).

    Returns:
        A DataFrame containing all combined features.
    """
    if fileName.lower().__contains__("train"):
        folderName = fileName.lower()
        fileName = "train"
    else:
        folderName = fileName.lower()
        fileName = "ind"

    col = ["label"]
    labels_path = os.path.join(
        folderName,
        "labels.csv",
    )
    labels = pd.read_csv(labels_path, header=0)
    combined_features = pd.DataFrame(labels)
    for i in range(0, len(features_list)):
        if features_list[i].upper() == "AAC":
            filepath = os.path.join(
                folderName,
                features_list[i].upper() + "_" + fileName + ".csv",
            )  # features/variable/train/PSTPP_train.csv
            aac_features = pd.read_csv(filepath, header=0)
            aac_features = aac_features.drop(columns=["label"])
            col.extend(aac_features.columns.tolist())
            if combined_features.shape[1] == 0:
                combined_features = aac_features
            else:
                combined_features = pd.concat([combined_features, aac_features], axis=1)

        if features_list[i].upper() == "DPC":
            filepath = os.path.join(
                folderName,
                features_list[i].upper() + "_" + fileName + ".csv",
            )
            _features = pd.read_csv(filepath, header=0)
            _features = _features.drop(columns=["label"])
            col.extend(_features.columns.tolist())
            if combined_features.shape[1] == 0:
                combined_features = _features
            else:
                combined_features = pd.concat([combined_features, _features], axis=1)

        if features_list[i].upper() == "PAAC":
            filepath = os.path.join(
                folderName,
                features_list[i].upper() + "_" + fileName + ".csv",
            )
            _features = pd.read_csv(filepath, header=0)
            _features = _features.drop(columns=["label"])
            col.extend(_features.columns.tolist())
            if combined_features.shape[1] == 0:
                combined_features = _features
            else:
                combined_features = pd.concat([combined_features, _features], axis=1)

        if features_list[i].upper() == "APAAC":
            filepath = os.path.join(
                folderName,
                features_list[i].upper() + "_" + fileName + ".csv",
            )
            _features = pd.read_csv(filepath, header=0)
            _features = _features.drop(columns=["label"])
            col.extend(_features.columns.tolist())
            if combined_features.shape[1] == 0:
                combined_features = _features
            else:
                combined_features = pd.concat([combined_features, _features], axis=1)

        if features_list[i].lower() == "pcp":
            filepath = os.path.join(
                folderName,
                features_list[i].upper() + "_" + fileName + ".csv",
            )
            _features = pd.read_csv(filepath, header=0)
            _features = _features.drop(columns=["label"])
            col.extend(_features.columns.tolist())
            if combined_features.shape[1] == 0:
                combined_features = _features
            else:
                combined_features = pd.concat([combined_features, _features], axis=1)

        if features_list[i].lower() == "onehot":
            filepath = os.path.join(
                folderName,
                features_list[i].upper() + "_" + fileName + ".csv",
            )
            _features = pd.read_csv(filepath, header=0)
            _features = _features.drop(columns=["label"])
            col.extend(_features.columns.tolist())
            if combined_features.shape[1] == 0:
                combined_features = _features
            else:
                combined_features = pd.concat([combined_features, _features], axis=1)

        if features_list[i].lower() == "esm2":
            filepath = os.path.join(
                folderName,
                features_list[i].upper() + "_" + fileName + ".csv",
            )
            esm2_features = pd.read_csv(filepath, header=0)
            esm2_features = esm2_features.drop(columns=["label"])
            col.extend(esm2_features.columns.tolist())
            if combined_features.shape[1] == 0:
                combined_features = esm2_features
            else:
                combined_features = pd.concat(
                    [combined_features, esm2_features], axis=1
                )

        if features_list[i].lower() == "pstpp":
            filepath = os.path.join(
                folderName,
                features_list[i].upper() + "_" + fileName + ".csv",
            )
            pstpp_features = pd.read_csv(filepath, header=0)
            pstpp_features = pstpp_features.drop(columns=["label"])
            col.extend(pstpp_features.columns.tolist())
            if combined_features.shape[1] == 0:
                combined_features = pstpp_features
            else:
                combined_features = pd.concat(
                    [combined_features, pstpp_features], axis=1
                )

        if features_list[i].lower() == "bert":
            filepath = os.path.join(
                folderName,
                features_list[i].upper() + "_" + fileName + ".csv",
            )
            bert_features = pd.read_csv(filepath, header=0)
            col.extend(bert_features.columns.tolist())
            if combined_features.shape[1] == 0:
                combined_features = bert_features
            else:
                # bert_features = bert_features.drop(columns=['label'])
                combined_features = pd.concat(
                    [combined_features, bert_features], axis=1
                )

        if features_list[i].lower() == "pssm":
            filepath = os.path.join(
                folderName,
                features_list[i].upper() + "_" + fileName + ".csv",
            )
            pssm_features = pd.read_csv(filepath, header=0)
            pssm_features = pssm_features.drop(columns=["label"])
            col.extend(pssm_features.columns.tolist())
            if combined_features.shape[1] == 0:
                combined_features = pssm_features
            else:
                combined_features = pd.concat(
                    [combined_features, pssm_features], axis=1
                )

        combined_features.columns = col
    return combined_features


def best_thresholds(y_true, proba):
    y_true = np.asarray(y_true).astype(int).ravel()
    proba = np.asarray(proba).ravel()
    ts = np.linspace(0.01, 0.99, 99)
    best_f1, t_f1 = -1.0, 0.5
    best_mcc, t_mcc = -1.0, 0.5
    best_sens, t_sens = -1.0, 0.5
    best_spec, t_spec = -1.0, 0.5
    best_acc, t_acc = -1.0, 0.5
    best_auc, t_auc = -1.0, 0.5
    best_map, t_map = -1.0, 0.5
    for t in ts:
        pred = (proba >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, pred) if (pred.max() != pred.min()) else -1.0
        acc = accuracy_score(y_true, pred)
        sens = recall_score(y_true, pred)
        _auc = roc_auc_score(y_true, pred, average="weighted")
        MAP = average_precision_score(y_true, pred, average="weighted")
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[1, 0]).ravel()
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        if f1 > best_f1:
            best_f1, t_f1 = f1, t
        if mcc > best_mcc:
            best_mcc, t_mcc = mcc, t
        if acc > best_acc:
            best_acc, t_acc = acc, t
        if sens > best_sens:
            best_sens, t_sens = sens, t
        if spec > best_spec:
            best_spec, t_spec = spec, t
        if _auc > best_auc:
            best_auc, t_auc = _auc, t
        if MAP > best_map:
            best_map, t_map = MAP, t
    return {
        "t_f1": float(t_f1),
        "best_f1": float(best_f1),
        "t_mcc": float(t_mcc),
        "best_mcc": float(best_mcc),
        "t_acc": float(t_acc),
        "best_acc": float(best_acc),
        "t_spec": float(t_spec),
        "best_spec": float(best_spec),
        "t_sens": float(t_sens),
        "best_sens": float(best_sens),
        "t_auc": float(t_auc),
        "best_auc": float(best_auc),
        "t_map": float(t_map),
        "best_map": float(best_map),
    }


def _binary_metrics_from_probs(y_true, y_prob, cutoff=0.5):
    results = best_thresholds(y_true, y_prob)
    # cutoff = thrs["t_f1"]
    y_pred = (y_prob >= cutoff).astype(int)
    acc = results["best_acc"]  # accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred)  # results["best_sens"]  #
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) else 0.0  # results["best_spec"]  #

    f1 = results["best_f1"]  # f1_score(y_true, y_pred)
    mcc = results["best_mcc"]  # matthews_corrcoef(y_true, y_pred)
    try:
        aucv = results["best_auc"]  # roc_auc_score(y_true, y_prob)
        ap = results[
            "best_map"
        ]  # average_precision_score(y_true, y_prob, average="weighted")  # <- add new line
    except ValueError:
        aucv = float("nan")
        ap = float("nan")  # <- add new line
    return dict(
        ACC=acc, SENS=sens, SPEC=spec, F1=f1, mAP=ap, MCC=mcc, AUC=aucv, y_pred=y_pred
    )  # <- add new line


def _mcnemar_from_folds(y_true_folds, pred_ref_folds, pred_other_folds):
    """Aggregate McNemar across folds by summing n01/n10, then one test."""
    n01 = n10 = 0
    for y, pr, po in zip(y_true_folds, pred_ref_folds, pred_other_folds):
        ref_correct = pr == y
        other_correct = po == y
        n01 += int(np.sum(ref_correct & ~other_correct))  # ref correct, other wrong
        n10 += int(np.sum(~ref_correct & other_correct))  # ref wrong, other correct
    if (n01 + n10) == 0:
        return float("nan"), n01, n10
    exact = (n01 + n10) < 25
    res = mcnemar([[0, n01], [n10, 0]], exact=exact, correction=not exact)
    return float(res.pvalue), n01, n10


def _cv_eval_one_model(
    model_name, X, y, k=5, epochs=50, batch=32, cutoff=0.5, verbose=0, feat_name="ACC"
):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    metrics_per_fold, y_true_folds, y_proba_folds, y_pred_folds, val_idx_folds = (
        [],
        [],
        [],
        [],
        [],
    )

    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        print(f"Fold {fold} on {model_name}")
        Xtr_raw, Xva_raw = X[tr], X[va]
        ytr, yva = y[tr].astype(int), y[va].astype(int)

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr_raw)
        Xva = scaler.transform(Xva_raw)

        model = MODEL_BUILDERS[model_name](Xtr.shape[1])

        reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            mode="max",
            factor=0.5,
            patience=8,
            verbose=0,
            min_lr=2e-5,
        )
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=20,
            restore_best_weights=True,
            verbose=0,
        )
        model_output_dir = "models_kfold"
        os.makedirs(model_output_dir, exist_ok=True)
        file_path = os.path.join(
            model_output_dir, f"{feat_name}_{model_name}.weights.h5"
        )
        save_best_model = tf.keras.callbacks.ModelCheckpoint(
            file_path,
            monitor="val_accuracy",
            verbose=1,
            save_best_only=True,
            mode="max",
            save_weights_only=True,
        )

        with tf.device("/GPU:0"):
            train_dataset = (
                tf.data.Dataset.from_tensor_slices((Xtr, ytr))
                .shuffle(1000)
                .map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.int32)))
                .batch(32)
                .prefetch(tf.data.AUTOTUNE)
            )
            test_dataset = (
                tf.data.Dataset.from_tensor_slices((Xva, yva))
                .map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.int32)))
                .batch(32)
                .prefetch(tf.data.AUTOTUNE)
            )

        t0 = time.perf_counter()
        history = model.fit(
            train_dataset,
            epochs=epochs,
            batch_size=batch,
            verbose=verbose,
            validation_data=test_dataset,
            callbacks=[reduce_on_plateau, early_stop, save_best_model],
        )
        train_seconds = time.perf_counter() - t0
        model.load_weights(file_path)

        new_file_path = file_path.replace(".weights.h5", ".h5")
        model.save(new_file_path)

        prob = model.predict(Xva, verbose=0).ravel()
        # If met["y_pred"] already exists, you can use it; otherwise define here:
        y_pred = (prob >= cutoff).astype(int)

        met = _binary_metrics_from_probs(yva, prob, cutoff=cutoff)

        fold_metrics = {
            k: met[k] for k in ("ACC", "SENS", "SPEC", "F1", "mAP", "MCC", "AUC")
        }
        fold_metrics["TIME_SEC"] = train_seconds
        fold_metrics["TIME_PER_EPOCH_SEC"] = train_seconds / len(
            history.history["loss"]
        )
        metrics_per_fold.append(fold_metrics)

        # Keep 1-D arrays; no .T
        y_true_folds.append(yva.copy())
        y_proba_folds.append(prob.copy())
        y_pred_folds.append(y_pred)  # or met["y_pred"].astype(int)
        val_idx_folds.append(va.copy())  # indices into original X

        tf.keras.backend.clear_session()

    return metrics_per_fold, y_true_folds, y_pred_folds, y_proba_folds, val_idx_folds


# ========== PIPELINE FUNCTION ==========
def train_val_pipeline(
    feature_list,
    fasta_list,
    models_list,
    combine_features=False,
    output_dir="results",
    cv_splits=5,
    reference_model="CNN",
    epochs=50,
    batch_size=32,
    cutoff=0.5,
    verbose=0,
):

    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)
    results = []  ### rename to results_train

    train_fasta = fasta_list[0]

    file_name = os.path.splitext(train_fasta)[0].split("/")[-1]

    for feat_name in feature_list:
        # Define file paths for training and independent feature files
        print(f"\n=== Feature Set: {feat_name} ===")

        # Load features
        if not combine_features:
            fv_set = encode_features(
                [feat_name],
                type="",
                fileName=train_fasta.lower(),
            )
        else:
            new_fea_list = []
            # new_fea_list.append(_labels)
            new_fea_list.extend(feature_list)
            # new_fea_list.remove(feat_name)
            fv_set = encode_features(new_fea_list, type="", fileName=file_name.lower())

        print(fv_set.head(10))
        # Build raw X,y and handle NaNs once per feature
        fv = fv_set.copy()
        fv = fv.fillna(0)
        y_all = fv["label"].astype(int).values
        X_all = fv.drop(columns=["label"]).values

        # store CV outputs per model (for McNemar)
        cv_metrics_by_model = {}  # name -> list of dicts per fold
        y_true_folds_any = None  # will take from first model run

        preds_by_model = {}
        proba_by_model = {}
        y_true_by_model = {}
        va_idx_by_model = {}

        for model_name in models_list:
            print(f"CV training {model_name} on {feat_name}...")

            met_folds, y_true_folds, y_pred_folds, y_proba_folds, val_idx_folds = (
                _cv_eval_one_model(
                    model_name,
                    X_all,
                    y_all,
                    k=cv_splits,
                    epochs=epochs,
                    batch=batch_size,
                    cutoff=cutoff,
                    verbose=verbose,
                    feat_name=feat_name,
                )
            )
            cv_metrics_by_model[model_name] = met_folds
            preds_by_model[model_name] = y_pred_folds
            proba_by_model[model_name] = y_proba_folds
            y_true_by_model[model_name] = y_true_folds
            va_idx_by_model[model_name] = val_idx_folds

            if y_true_folds_any is None:
                y_true_folds_any = y_true_folds

        # === Aggregate per model: mean ± std (Table 1 style)
        rows = []
        for model_name in models_list:
            mlist = cv_metrics_by_model[model_name]
            ACC = np.array([m["ACC"] for m in mlist])
            SENS = np.array([m["SENS"] for m in mlist])
            SPEC = np.array([m["SPEC"] for m in mlist])
            F1 = np.array([m["F1"] for m in mlist])
            MAP = np.array([m["mAP"] for m in mlist])  # <- add new line
            MCC = np.array([m["MCC"] for m in mlist])
            AUCV = np.array([m["AUC"] for m in mlist])
            TSEC = np.array([m["TIME_SEC"] for m in mlist])
            TPE = np.array([m["TIME_PER_EPOCH_SEC"] for m in mlist])

            row = dict(
                Feature_Set=feat_name,
                Model=model_name,
                ACC_Mean=ACC.mean(),
                ACC_Std=ACC.std(ddof=1),
                Sens_Mean=SENS.mean(),
                Sens_Std=SENS.std(ddof=1),
                Spec_Mean=SPEC.mean(),
                Spec_Std=SPEC.std(ddof=1),
                F1_Mean=F1.mean(),
                F1_Std=F1.std(ddof=1),
                mAP_MEAN=MAP.mean(),  # <- add new line
                mAP_STD=MAP.std(ddof=1),  # <- add new line
                MCC_Mean=MCC.mean(),
                MCC_Std=MCC.std(ddof=1),
                AUC_Mean=AUCV.mean(),
                AUC_Std=AUCV.std(ddof=1),
                # NEW timing summaries
                TrainTime_MeanSec=TSEC.mean(),
                TrainTime_StdSec=TSEC.std(ddof=1),
                TrainTime_PerEpoch_MeanSec=TPE.mean(),
                TrainTime_PerEpoch_StdSec=TPE.std(ddof=1),
            )
            rows.append(row)

        # === McNemar P-values vs reference model (aggregate across folds)
        ref = reference_model
        if ref not in preds_by_model:
            print(
                f"[WARN] reference_model='{ref}' not in models_list; skipping McNemar for {feat_name}"
            )
        else:
            for row in rows:
                model_name = row["Model"]
                if model_name == ref:
                    row["McNemar_P"] = np.nan
                    row["McNemar_n01"] = np.nan
                    row["McNemar_n10"] = np.nan
                else:
                    p, n01, n10 = _mcnemar_from_folds(
                        y_true_folds_any,
                        preds_by_model[ref],
                        preds_by_model[model_name],
                    )
                    row["McNemar_P"] = p
                    row["McNemar_n01"] = n01
                    row["McNemar_n10"] = n10

        # append to global results
        results.extend(rows)

        # === Save per-sample CV outputs in a tidy table ===
        rows_long = []
        for model_name in models_list:
            # each is a list of k arrays
            ytf_list = y_true_by_model[model_name]
            ypf_list = proba_by_model[model_name]
            ypred_list = preds_by_model[model_name]
            idx_list = va_idx_by_model[model_name]

            for fold_id, (idxs, yt, yp, yhat) in enumerate(
                zip(idx_list, ytf_list, ypf_list, ypred_list), start=1
            ):
                # sanity: all same length
                n = len(yt)
                assert len(yp) == n and len(yhat) == n and len(idxs) == n
                rows_long.append(
                    pd.DataFrame(
                        {
                            "Feature_Set": feat_name,
                            "Model": model_name,
                            "Fold": fold_id,
                            "orig_idx": idxs,  # position in original X_all
                            "y_true": yt.astype(int),
                            "proba": yp.astype(float),
                            "y_pred": yhat.astype(int),
                        }
                    )
                )

        cv_outputs_df = pd.concat(rows_long, ignore_index=True)
        os.makedirs(output_dir, exist_ok=True)
        out_long_path = os.path.join(output_dir, f"CV_outputs_{feat_name}.csv")
        cv_outputs_df.to_csv(out_long_path, index=False)
        print(f"Saved per-sample CV outputs for {feat_name} -> {out_long_path}")

        if combine_features:
            break

    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(output_dir, "Table1_CV_all_features.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nAll-feature CV Table‑1 saved to {results_csv_path}")
    return results_df


def main(
    feature_list,
    train_fasta_list,
    models_list,
    combine_features: bool = False,
):

    results_df = train_val_pipeline(
        feature_list,
        train_fasta_list,
        models_list,
        combine_features=combine_features,
        output_dir="results_kfold",
        cv_splits=2,  # <-- K-folds
        reference_model="CNN",  # <-- choose your reference (e.g., 'CNN' or 'FCNN')
        epochs=5,
        batch_size=32,
        cutoff=0.5,
        verbose=1,
    )
    print(results_df)


if __name__ == "__main__":
    # Feature sets to use
    feature_list = [
        # "AAC",
        # "APAAC",
        # "DPC",
        # "ONEHOT",
        # "PAAC",
        # "PCP",
        # "PSSM",
        "PSTPP",
    ]
    train_fasta_list = ["features/train_ibce"]

    models_list = ["CNN", "FCNN", "RNN", "GRU", "LSTM"]

    main(
        feature_list,
        train_fasta_list,
        models_list,
        combine_features=False,
    )
