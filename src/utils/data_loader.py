import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def get_binary_dataset_names():
    return [
        'Coffee',
        'Computers'
    ]

def get_four_dataset_names():
    return [
        'Car'
    ]

def get_multiclass_dataset_name():
    return [
        'EOGHorizontalSignal'
    ]


def load_dataset(ds_name):
    LE = LabelEncoder()
    ds_train = pd.read_table(f'data/UCR_Archive/{ds_name}/{ds_name}_TRAIN.tsv', header=None).to_numpy()
    X_train = ds_train[:, 1:]
    Y_train = ds_train[:, 0]
    Y_train = LE.fit_transform(Y_train)

    ds_test = pd.read_table(f'data/UCR_Archive/{ds_name}/{ds_name}_TEST.tsv', header=None).to_numpy()
    X_test = ds_test[:, 1:]
    Y_test = ds_test[:, 0]
    Y_test = LE.transform(Y_test)

    X_train = X_train[:, np.newaxis, :]
    X_test = X_test[:, np.newaxis, :]

    return X_train, Y_train, X_test, Y_test
