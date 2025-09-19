from hmac import new
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics

import itertools
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA, TruncatedSVD, IncrementalPCA, KernelPCA
from sklearn.preprocessing import OneHotEncoder


from sklearn.feature_selection import SelectKBest
from scipy.stats import f

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

import math, time, random, datetime
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.metrics import r2_score, accuracy_score, classification_report
from sklearn.metrics import roc_curve, auc

from tqdm import tqdm

from Bio import SeqIO
from propy import PyPro
from modlamp.descriptors import GlobalDescriptor
import csv
import pickle
import sys

import tensorflow as tf

import joblib
import itertools
from sklearn.feature_selection import chi2

# from prediction_independent import combine_features

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


def calculate_performace(test_num, y_prob, y_test):

    y_pred = (y_prob >= 0.5).astype(int)
    # Metrics
    _f1 = f1_score(y_test, y_pred)
    _acc = accuracy_score(y_test, y_pred)
    _auc = roc_auc_score(y_test, y_prob)
    _mcc = matthews_corrcoef(y_test, y_pred)

    _cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = _cm.ravel()

    # Calculate Sensitivity
    sens = TP / (TP + FN)

    # Calculate Specificity
    spec = TN / (TN + FP)

    acc = _acc  # float(tp + tn) / test_num
    precision = float(TP) / (TP + FP)
    sensitivity = sens  # float(tp) / (tp + fn)
    specificity = spec  # float(tn) / (tn + fp)
    MCC = _mcc  # (float(tp) * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return acc, precision, sensitivity, specificity, MCC, _f1, _auc, _cm


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


def readAAP(file):
    """
    read AAP features from the AAP textfile
    """
    try:
        aapdic = {}
        aapdata = open(file, "r")
        for l in aapdata.readlines():
            aapdic[l.split()[0]] = float(l.split()[1])
        aapdata.close()
        return aapdic
    except:
        print(
            "Error in reading AAP feature file. Please make sure that the AAP file is correctly formatted"
        )
        sys.exit()


# read AAT features from the AAT textfile
def readAAT(file):
    """
    read AAT features from the AAT textfile
    """
    try:
        aatdic = {}
        aatdata = open(file, "r")
        for l in aatdata.readlines():
            aatdic[l.split()[0][0:3]] = float(l.split()[1])
        aatdata.close()
        return aatdic
    except:
        print(
            "Error in reading AAT feature file. Please make sure that the AAT file is correctly formatted"
        )
        sys.exit()


def AAT(sequence):
    """
    Calculates a placeholder Amino Acid Trimer (AAT) antigenicity score
    for a protein sequence.

    Args:
        sequence: A string representing the protein sequence.

    Returns:
        The calculated placeholder AAT antigenicity score.
    """
    # Placeholder AAT scale (replace with actual scale from text when available)
    # This is a hypothetical example. The actual scale would be a dictionary
    # mapping amino acid trimers (e.g., "AGC") to their antigenicity contribution.
    # For now, we'll just sum some arbitrary values based on the amino acids.
    aatdic = readAAT("aat-general.txt.normal")
    aat_score = 0
    count = 0
    for i in range(len(sequence) - 2):
        try:
            trimer = aatdic[sequence[i : i + 3]]
            # In a real scenario, you would look up the trimer in a dictionary
            # and add its predefined antigenicity value to the score.
            # Example placeholder calculation:
            # if len(trimer) == 3:
            aat_score += np.round(
                trimer, 4
            )  # ord(trimer[0]) + ord(trimer[1]) + ord(trimer[2])
            count += 1
        except KeyError:
            aat_score += float(-1)
            count += 1
    if count > 0:
        aat_score = np.round((aat_score / count), 4)

    return aat_score


# Example usage with a dummy sequence
# dummy_sequence = "AGCEDF"
# aat_score = calculate_aat_antigenicity(dummy_sequence)
# print(f"AAT antigenicity score for {dummy_sequence}: {aat_score}")


def AAP(sequence):
    """
    Calculates the Amino Acid Pair (AAP) antigenicity score for a protein sequence.

    Args:
        sequence: A string representing the protein sequence.

    Returns:
        The calculated AAP antigenicity score.
    """
    # Placeholder AAP scale (replace with actual scale from text when available)
    # This is a hypothetical example. The actual scale would be a dictionary
    # mapping amino acid pairs (e.g., "AG") to their antigenicity contribution.
    # For now, we'll just sum some arbitrary values based on the amino acids.
    # A more realistic placeholder would involve a lookup table.

    # Let's use a simple example: sum the ASCII values of the pair.
    # This is NOT a real antigenicity scale, just a placeholder for demonstration.
    aapdic = readAAP("aap-general.txt.normal")
    aap_score = 0
    count = 0
    for i in range(len(sequence) - 1):
        try:
            pair = aapdic[sequence[i : i + 2]]
            # In a real scenario, you would look up the pair in a dictionary
            # and add its predefined antigenicity value to the score.
            # Example placeholder calculation:
            # if len(pair) == 2:
            aap_score += round(pair, 4)  # ord(pair[0]) + ord(pair[1])
            count += 1
        except KeyError:
            aap_score += float(-1)
            count += 1
    if count > 0:
        aap_score = np.round((aap_score / count), 4)

    return aap_score


# Dipeptide Composition feature


def DPC(sequence):
    """
    Dipeptide Composition feature
    """
    feature = []
    for seq in sequence:
        seq = seq.split("\t")[0]
        dpc = PyPro.CalculateDipeptideComposition(seq)
        feature.append(list(dpc.values()))
        name = list(dpc.keys())
    return feature, name


def AAC(sequence):
    """
    Dipeptide Composition feature
    """
    feature = []
    for seq in sequence:
        seq = seq.split("\t")[0]
        aac = PyPro.CalculateAAComposition(seq)
        feature.append(list(aac.values()))
        name = list(aac.keys())
    return feature, name


def PAAC(sequence):
    """
    PAAC Composition feature
    """
    feature = []
    for seq in sequence:
        seq = seq.split("\t")[0]
        DesObject = PyPro.GetProDes(seq.upper())
        paac = DesObject.GetPAAC(lamda=5, weight=0.05)
        feature.append(list(paac.values()))
        name = list(paac.keys())
    return feature, name


def APAAC(sequence):
    """
    PAAC Composition feature
    """
    feature = []
    for seq in sequence:
        seq = seq.split("\t")[0]
        DesObject = PyPro.GetProDes(seq.upper())
        paac = DesObject.GetAPAAC(lamda=5, weight=0.05)
        feature.append(list(paac.values()))
        name = list(paac.keys())
    return feature, name


def pcp(sequence):
    """
    Physio-chemical feature
    """
    feature = []

    for seq in sequence:
        name = []
        seq = seq.split("\t")[0]
        desc = GlobalDescriptor(seq.upper())
        desc.calculate_all()
        feature.append(desc.descriptor.flatten()[1:].tolist())
        name = list(desc.featurenames[1:])
    return feature, name


aa_list = list("ACDEFGHIKLMNPQRSTVWYX")
aa_dict = {aa: idx for idx, aa in enumerate(aa_list)}


def one_hot_encoder(seq, maxlen=20):
    mat = np.zeros((maxlen, len(aa_list)))
    for i, aa in enumerate(seq[:maxlen]):
        if aa in aa_dict:
            mat[i, aa_dict[aa]] = 1
    return mat.flatten()


def one_hot_encode_sequences(sequences, maxlen=20):
    """
    One-hot encodes a list of sequences.

    Args:
        sequences: A list of protein sequences.
        maxlen: Maximum length of the sequence to encode.

    Returns:
        A 2D numpy array where each row is the one-hot encoded representation of a sequence.
    """
    sequences = adjust_sequences(sequences)
    maxlen = 25
    return np.array([one_hot_encoder(seq, maxlen) for seq in sequences])


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
            )  # features/variable/train/PSTPP_train.csv
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
            # filepath = os.path.join('features', 'esm2', type, fileName)
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
        if features_list[i].lower() == "esm21ht":
            # filepath = os.path.join('features', 'esm2', type, fileName)
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
        if (
            features_list[i].lower() == "pstpp"
        ):  # features/fixed/fixed_all_bcell_cd/PSSTP_fixed_all_bcell_cd.csv
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
            # filepath = os.path.join('features', 'protBert', type, fileName)
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
            # filepath = os.path.join('features', 'pssm', type, fileName)
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


def _cv_scores_for_model(
    model_name,
    model_builder,
    X,
    y,
    callbacks,
    cv_splits=5,
    epochs=10,
    ep_interval=1,
    cutoff=0.5,
    fit_kwargs=None,
):
    """
    Train *fresh* model instances per fold, collect F1 and MCC.
    Returns: f1_list, mcc_list
    """
    if fit_kwargs is None:
        fit_kwargs = {}
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    f1s, mccs = [], []

    for tr_idx, va_idx in skf.split(X, y):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        fea_dim = Xtr.shape[1]

        # build a fresh model for the fold
        if model_name == "FCNN":
            model = build_fcnn(fea_dim)
        elif model_name == "RNN":
            model = build_rnn(fea_dim)
        elif model_name == "GRU":
            model = build_gru(fea_dim)
        elif model_name == "LSTM":
            model = build_lstm(fea_dim)
        elif model_name == "CNN":
            model = build_cnn(fea_dim)
        else:
            model = model_builder(fea_dim)  # fallback

        # match input shapes for validation set
        if model_name in ("GRU", "LSTM", "RNN"):
            Xtr_fit = np.asarray(Xtr).reshape(-1, 1, fea_dim)
            Xva_eval = np.asarray(Xva).reshape(-1, 1, fea_dim)
        elif model_name in ("CNN", "CNN_LSTM"):
            Xtr_fit = np.asarray(Xtr).reshape(-1, fea_dim, 1)
            Xva_eval = np.asarray(Xva).reshape(-1, fea_dim, 1)
        else:
            Xtr_fit = Xtr
            Xva_eval = Xva

        # minimal callbacks to keep CV quick/stable
        history = model.fit(
            Xtr_fit,
            ytr,
            epochs=epochs,
            batch_size=32,
            verbose=0,
            validation_data=(Xva_eval, yva),
            callbacks=[
                # SelectiveProgbarLogger(verbose=0, epoch_interval=ep_interval),
                callbacks,
            ],
            **fit_kwargs,
        )

        proba = model.predict(Xva_eval, verbose=0).ravel()
        yhat = (proba >= cutoff).astype(int)

        f1s.append(f1_score(y_true=yva, y_pred=yhat))
        mccs.append(matthews_corrcoef(yva, yhat))

    return f1s, mccs


def _holm_correction(pvals, labels):
    """
    Holm-Bonferroni step-down correction.
    Returns dict[label] -> adjusted_p
    """
    idx_sorted = np.argsort(pvals)
    m = len(pvals)
    adjusted = [None] * m
    for rank, i in enumerate(idx_sorted):
        adjusted[i] = min(1.0, (m - rank) * pvals[i])
    # enforce monotonicity (optional)
    for i in range(1, m):
        j1, j2 = idx_sorted[i - 1], idx_sorted[i]
        adjusted[j2] = max(adjusted[j2], adjusted[j1])
    return {labels[i]: adjusted[i] for i in range(m)}


def train_and_test(fv_set, model_name, feat_name, callbacks):
    # print(fv_set.head(10))
    results = []
    X_train, X_test, y_train, y_test = train_test_split(
        fv_set.drop(columns=["label"]),
        fv_set["label"],
        test_size=0.2,
        random_state=245,
        shuffle=True,
        stratify=fv_set["label"],
    )

    if X_train.isnull().values.any():
        count_nan = X_train.isnull().sum().sum()
        print(f"X_train contains {count_nan} NaN values. Filling NaN values with 0.")
        X_train = X_train.fillna(0)
    if X_test.isnull().values.any():
        count_nan = X_test.isnull().sum().sum()
        print(f"X_test contains {count_nan} NaN values. Filling NaN values with 0.")
        X_test = X_test.fillna(0)

    # Ensure labels are integers
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # Label encode the labels
    y_train = LabelEncoder().fit_transform(y_train)
    y_test = LabelEncoder().fit_transform(y_test)
    # Print shapes
    print(f"Training set shape: {X_train.shape}, Labels shape: {y_train.shape}")
    print(f"Validation set shape: {X_test.shape}, Labels shape: {y_test.shape}")

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    # Build model
    model = MODEL_BUILDERS[model_name](X_train.shape[1])

    # Train
    model.fit(
        train_dataset.batch(32),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
        validation_data=test_dataset.batch(32),
    )

    # Predict
    y_pred_prob = model.predict(X_test).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Metrics
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    mcc = matthews_corrcoef(y_test, y_pred)

    # Save results
    results.append(
        {
            "Feature_Set": feat_name,
            "Model": model_name,
            "F1_Score": f1,
            "Accuracy": acc,
            "AuROC": auc,
            "MCC": mcc,
        }
    )


# ========== PIPELINE FUNCTION ==========
def train_val_pipeline(
    feature_list,
    fasta_list,
    models_list,
    combine_features=False,
    output_dir="results",
    epochs=50,
    batch_size=32,
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
        X_train, X_test, y_train, y_test = train_test_split(
            fv_set.drop(columns=["label"]),
            fv_set["label"],
            test_size=0.2,
            random_state=245,
            shuffle=True,
            stratify=fv_set["label"],
        )

        if X_train.isnull().values.any():
            count_nan = X_train.isnull().sum().sum()
            print(
                f"X_train contains {count_nan} NaN values. Filling NaN values with 0."
            )
            X_train = X_train.fillna(0)
        if X_test.isnull().values.any():
            count_nan = X_test.isnull().sum().sum()
            print(f"X_test contains {count_nan} NaN values. Filling NaN values with 0.")
            X_test = X_test.fillna(0)

        # Ensure labels are integers
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        # Label encode the labels
        y_train = LabelEncoder().fit_transform(y_train)
        y_test = LabelEncoder().fit_transform(y_test)
        # Print shapes
        print(f"Training set shape: {X_train.shape}, Labels shape: {y_train.shape}")
        print(f"Validation set shape: {X_test.shape}, Labels shape: {y_test.shape}")

        # Scale features
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        from sklearn.utils import compute_class_weight

        classes = np.array([0, 1])
        class_weights = compute_class_weight(
            class_weight="balanced", classes=classes, y=y_train
        )
        class_weight_dict = dict(zip(classes, class_weights))
        print("Class weights:", class_weight_dict)

        for model_name in models_list:
            print(f"Training {model_name} on {feat_name}...")
            # if model_name.upper() == "CNN":
            #     # tf.config.set_visible_devices([], "GPU")
            #     X_train = X_train.reshape(-1, X_train.shape[1], 1)
            #     X_test = X_test.reshape(-1, X_test.shape[1], 1)
            #     # Build model
            #     print(X_train.shape[1:])
            #     model = MODEL_BUILDERS[model_name](X_train.shape[1])
            # elif model_name.upper() in ["RNN", "LSTM", "GRU"]:
            #     X_train = X_train.reshape(-1, 1, X_train.shape[1])
            #     X_test = X_test.reshape(-1, 1, X_test.shape[1])
            #     # Build model
            #     print(X_train.shape[1:])
            #     model = MODEL_BUILDERS[model_name](X_train.shape[2])
            #     # tf.config.set_visible_devices([], "GPU")
            model = MODEL_BUILDERS[model_name](X_train.shape[1])
            # joblib.dump(scaler, f"{model_name}_{feat_name}_scaler.pkl")
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
            file_path = os.path.join("models", model_name, f"{feat_name}_{model_name}")
            # file_path = f"models/{model_name}_{feat_name}/saved_models/"
            checkpoint = tf.keras.callbacks.ModelCheckpoint(
                file_path,
                monitor="val_accuracy",
                verbose=1,
                save_best_only=True,
                mode="max",
                save_weights_only=False,
            )
            reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_accuracy",
                mode="max",
                factor=0.5,
                patience=10,
                verbose=1,
                min_lr=0.00002,
            )
            restore_best_weights = tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=50,
                restore_best_weights=True,
                verbose=1,
                mode="max",
            )
            callbacks_list = [checkpoint, reduce_on_plateau, restore_best_weights]
            # Train
            model.fit(
                train_dataset.batch(batch_size),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks_list,
                verbose=1,
                validation_data=test_dataset.batch(batch_size),
                class_weight=class_weight_dict,
            )

            # Predict
            y_pred_prob = model.predict(X_test).ravel()
            y_pred = (y_pred_prob >= 0.5).astype(int)

            # Metrics
            f1 = f1_score(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)
            _auc = roc_auc_score(y_test, y_pred_prob)
            mAP = average_precision_score(y_test, y_pred_prob)
            mcc = matthews_corrcoef(y_test, y_pred)

            cm = confusion_matrix(y_test, y_pred)
            TN, FP, FN, TP = cm.ravel()

            # Calculate Sensitivity
            sens = TP / (TP + FN)

            # Calculate Specificity
            spec = TN / (TN + FP)

            # Save results
            results.append(
                {
                    # "Feature_Set": feat_name,
                    "Model": model_name,
                    "Accuracy": acc,
                    "F1_Score": f1,
                    "Sensitivity": sens,
                    "Specificity": spec,
                    "mAP": mAP,
                    "AuROC": _auc,
                    "MCC": mcc,
                }
            )

            # Save model
            if combine_features:
                feat_name = "combined"

            model_save_path = os.path.join(
                "models", model_name, f"{feat_name}_{model_name}.h5"
            )
            model.save(model_save_path)

            # Save scaler for reproducibility
            scaler_path = os.path.join(
                "models", model_name, f"{feat_name}_{model_name}_scaler.pkl"
            )
            joblib.dump(scaler, scaler_path)

            print(f"Model saved to {model_save_path}", f"Scaler saved to {scaler_path}")

        if combine_features:
            break

    # Save results CSV
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(output_dir, f"{model_name}_val_results.csv")
    results_df.to_csv(results_csv_path, index=False)

    print(f"\nResults saved to {results_csv_path}")
    return results_df


def predict_pipeline(
    feature_list, fasta_list, models_list, combine_features=False, output_dir="results"
):
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)
    results = []  ### rename to results_train
    model_preds = []
    train_fasta = fasta_list[0]

    file_name = os.path.splitext(train_fasta)[0].split("/")[-1]

    for fasta in fasta_list:
        if fasta.lower().__contains__("train"):
            continue
        else:
            ind_fasta = fasta

        file_name = os.path.splitext(ind_fasta)[0].split("/")[-1]
        for feat_name in feature_list:
            # Define file paths for training and independent feature files
            print(f"\n=== Feature Set: {feat_name}_{file_name} ===")

            # Load features
            if not combine_features:
                fv_set = encode_features(
                    [feat_name],
                    type="",
                    fileName=ind_fasta.lower(),
                )
            else:
                new_fea_list = []
                # new_fea_list.append(_labels)
                new_fea_list.extend(feature_list)
                # new_fea_list.remove(feat_name)
                fv_set = encode_features(
                    new_fea_list, type="", fileName=file_name.lower()
                )

            for model_name in models_list:
                print(f"Training {model_name} on {feat_name}...")
                # print(fv_set.head(10))
                X_test, y_test = fv_set.drop(columns=["label"]), fv_set["label"]

                if X_test.isnull().values.any():
                    count_nan = X_test.isnull().sum().sum()
                    print(
                        f"X_test contains {count_nan} NaN values. Filling NaN values with 0."
                    )
                    X_test = X_test.fillna(0)

                # Ensure labels are integers
                y_test = y_test.astype(int)

                # Label encode the labels
                y_test = LabelEncoder().fit_transform(y_test)

                # Load Scalar to Scale features
                scaler_path = os.path.join(
                    "models", model_name, f"{feat_name}_{model_name}_scaler.pkl"
                )
                scaler = joblib.load(scaler_path)
                print(
                    f"Loaded scaler from {scaler_path} shape of X_test: {X_test.shape}"
                )
                X_test = scaler.transform(X_test)

                print(
                    f"Validation set shape: {X_test.shape}, Labels shape: {y_test.shape}"
                )

                # Load model
                file_path = os.path.join(
                    "models", model_name, f"{feat_name}_{model_name}"
                )
                model = tf.keras.models.load_model(file_path)
                # model = model.load_weights(file_path)

                # Predict
                y_pred_prob = model.predict(X_test).ravel()
                if file_name.lower().__contains__("abcpred"):
                    y_pred = (y_pred_prob >= 0.77).astype(int)
                if file_name.lower().__contains__("ibce"):
                    y_pred = (y_pred_prob >= 0.47).astype(int)
                elif file_name.lower().__contains__("clbe"):
                    y_pred = (y_pred_prob >= 0.56).astype(int)
                else:
                    y_pred = (y_pred_prob >= 0.5).astype(int)
                # Metrics
                f1 = f1_score(y_test, y_pred)
                acc = accuracy_score(
                    y_test,
                    y_pred,
                )
                _auc = roc_auc_score(y_test, y_pred_prob)
                mAP = average_precision_score(y_test, y_pred_prob)
                mcc = matthews_corrcoef(y_test, y_pred)

                cm = confusion_matrix(y_test, y_pred)

                TN, FP, FN, TP = cm.ravel()

                # Calculate Sensitivity
                sens = TP / (TP + FN)

                # Calculate Specificity
                spec = TN / (TN + FP)

                # Save results
                model_preds.append(
                    pd.DataFrame(
                        {
                            "Model": model_name,
                            "Dataset": file_name,
                            "orig_idx": fv_set.index,  # position in original X_all
                            "y_true": y_test.astype(int).ravel(),
                            "proba": y_pred_prob.astype(float).ravel(),
                            "y_pred": y_pred.astype(int).ravel(),
                        }
                    )
                )

                # Save results
                results.append(
                    {
                        # "Feature_Set": feat_name,
                        "Model": model_name,
                        "Dataset": file_name,
                        "Accuracy": acc,
                        "F1_Score": f1,
                        "Sensitivity": sens,
                        "Specificity": spec,
                        "mAP": mAP,
                        "AuROC": _auc,
                        "MCC": mcc,
                    }
                )

            if combine_features:
                break

    # # Save results CSV
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(
        output_dir, f"{feat_name}_{file_name}_ind_results.csv"
    )
    results_df.to_csv(results_csv_path, index=False)

    # Save results CSV
    model_preds_df = pd.concat(
        model_preds, ignore_index=True
    )  # pd.DataFrame(model_preds)
    results_csv_path = os.path.join(
        output_dir, f"{feat_name}_{file_name}_prediction.csv"
    )
    model_preds_df.to_csv(results_csv_path, index=False)

    print(f"\nResults saved to {results_csv_path}")
    return results_df


def main(
    feature_list,
    train_fasta_list,
    ind_fasta_list,
    models_list,
    training: bool = False,
    prediction: bool = True,
    combine_feature: bool = False,
    seq_type="variable",
):
    if training:
        results_df = train_val_pipeline(
            feature_list,
            train_fasta_list,
            models_list,
            combine_feature,
            output_dir="results",
            epochs=100,
            batch_size=32,
        )
        print(results_df)
    if prediction:
        predict_results_df = predict_pipeline(
            feature_list,
            ind_fasta_list,
            models_list,
            combine_feature,
            output_dir="results/compare_models",
        )
        print(predict_results_df)


if __name__ == "__main__":
    # iBCE_training_eval()
    feature_list = [
        "PSTPP",
        # "AAC",
        # "APAAC",
        # "DPC",
        # "ONEHOT",
        # "PAAC",
        # "PCP",
        # "PSSM",
    ]
    train_fasta_list = ["features/train_ibce"]

    ind_fasta_list = [
        "features/ind_ibce",
        "features/ind_clbe",
        "features/ind_abcpred",
    ]  # ["datasets/processed/variable/train/Train.fasta", "datasets/processed/variable/ind/Ind.fasta", "datasets/processed/variable/ind_lbtope/lbtope.fasta", "datasets/processed/variable/ind_abcpred/abcpred.fasta"]#,"datasets/processed/fixed/fixed_all_bcell_cd.fasta"]# ["datasets/iBCE-EL_training/Train-ibce.fasta", "datasets/iBCE-EL_training/Ind-ibce.fasta"] #["datasets/LBtope/lbtope_all_epitopes.fasta", "datasets/LBtope/LBtope_Fixed_ALL.fasta"] #["datasets/BCPreds/bcpred_all_20mer.fasta", "datasets/BCPreds/bcpred_independent_20mer.fasta"]
    models_list = ["CNN"]  # , "RNN", "GRU", "LSTM", "FCNN"]  #

    main(
        feature_list,
        train_fasta_list,
        ind_fasta_list,
        models_list,
        training=False,
        prediction=True,
        combine_feature=False,
    )
