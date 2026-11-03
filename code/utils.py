import numpy as np
import pandas as pd
import random
import torch
from sktime.transformations.panel.rocket import Rocket
from sklearn.preprocessing import LabelEncoder
from aeon.transformations.collection.convolution_based import HydraTransformer

# Better interface with helper functions
class RocketKernel:

    def __init__(self, seed, ts_length, ppv_only=True):
        self.rocket = Rocket(num_kernels=1, random_state=int(seed))
        self.rocket.fit(np.zeros((1, 1, ts_length)))
        self.ppv_only = ppv_only

    def transform(self, X):
        if len(X.shape) <= 1:
            X = X.reshape(1, -1)

        X_transformed = self.rocket.transform(np.expand_dims(X, 1)).to_numpy()
        if self.ppv_only:
            X_transformed = X_transformed[:, 0:1]

        return X_transformed


class HydraGroup:
    def __init__(self, seed, n_kernels, ts_length):
        self.hydra = HydraTransformer(n_kernels = n_kernels, n_groups = 1, random_state = int(seed))
        self.hydra.fit(np.zeros((1, 1, ts_length)))

    def transform(self, X):
        if len(X.shape) <= 1:
            X = X.reshape(1, -1)

        X_transformed = self.hydra.transform(np.expand_dims(X, 1))

        return X_transformed

def hydra_transform_seeds(X, seeds, n_kernels, ts_length):
    K = [HydraGroup(int(seed),n_kernels, ts_length) for seed in seeds]
    return np.concatenate([k.transform(X) for k in K], axis=1)

def transform_seeds(X, seeds, ts_length):
    K = [RocketKernel(int(seed), ts_length, ppv_only=True) for seed in seeds]
    return np.concatenate([k.transform(X) for k in K], axis=1)

class SparseScaler:
    """Sparse Scaler for hydra transform."""

    def __init__(self, mask=True, exponent=4):
        self.mask = mask
        self.exponent = exponent

    def fit(self, X, y=None):
        X = X.clamp(0).sqrt()

        self.epsilon = (X == 0).float().mean(0) ** self.exponent + 1e-8

        self.mu = X.mean(0)
        self.sigma = X.std(0) + self.epsilon

    def transform(self, X, y=None):
        X = X.clamp(0).sqrt()

        if self.mask:
            return ((X - self.mu) * (X != 0)) / self.sigma
        else:
            return (X - self.mu) / self.sigma

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


def split_data_dirichlet(train_features, train_targets, test_features, test_targets, num_clients, alpha=0.5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    num_classes = len(set(train_targets))
    client_data = {i: {'train': {'features': [], 'targets': []},
                      'test': {'features': [], 'targets': []}} for i in range(num_clients)}
    
    # Check if we have enough samples per class
    min_samples_per_class = num_clients  # Need at least one per client
    train_class_counts = np.bincount(train_targets)
    test_class_counts = np.bincount(test_targets)
    
    for c in range(num_classes):
        if train_class_counts[c] < min_samples_per_class:
            raise ValueError(f"Class {c} has only {train_class_counts[c]} train samples but needs at least {min_samples_per_class}")
        if test_class_counts[c] < min_samples_per_class:
            raise ValueError(f"Class {c} has only {test_class_counts[c]} test samples but needs at least {min_samples_per_class}")
    
    def get_class_indices(features, targets):
        return {i: np.where(np.array(targets) == i)[0] for i in range(num_classes)}
    
    train_class_indices = get_class_indices(train_features, train_targets)
    test_class_indices = get_class_indices(test_features, test_targets)

    class_proportions = {c: np.random.dirichlet(alpha * np.ones(num_clients)) for c in range(num_classes)}

    def assign_data(features, targets, class_indices, split_name):
        for c in range(num_classes):
            indices = class_indices[c]
            np.random.shuffle(indices)
            
            mandatory_assignments = indices[:num_clients]
            remaining_indices = indices[num_clients:]
            
            for i in range(num_clients):
                idx = mandatory_assignments[i]
                client_data[i][split_name]['features'].append(features[idx])
                client_data[i][split_name]['targets'].append(targets[idx])
            
            if len(remaining_indices) > 0:
                proportions = (np.cumsum(class_proportions[c]) * len(remaining_indices)).astype(int)[:-1]
                client_splits = np.split(remaining_indices, proportions)
                
                for i, split in enumerate(client_splits):
                    client_data[i][split_name]['features'].extend(features[idx] for idx in split)
                    client_data[i][split_name]['targets'].extend(targets[idx] for idx in split)
    
    assign_data(train_features, train_targets, train_class_indices, 'train')
    assign_data(test_features, test_targets, test_class_indices, 'test')
    
    for i in range(num_clients):
        for split in ['train', 'test']:
            client_data[i][split]['features'] = np.array(client_data[i][split]['features'])
            client_data[i][split]['targets'] = np.array(client_data[i][split]['targets'])
    
    return ([client_data[i]['train']['features'] for i in range(num_clients)],
            [client_data[i]['train']['targets'] for i in range(num_clients)],
            [client_data[i]['test']['features'] for i in range(num_clients)],
            [client_data[i]['test']['targets'] for i in range(num_clients)])




def get_binary_dataset_names():
    return [
        'BeetleFly',
        'BirdChicken',
        'Chinatown',
        'Coffee',
        'Computers',
        'DistalPhalanxOutlineCorrect',
        'DodgerLoopGame',
        'DodgerLoopWeekend',
        'Earthquakes',
        'ECG200',
        'ECGFiveDays',
        'FordA',
        'FordB',
        'FreezerRegularTrain',
        'FreezerSmallTrain',
        'GunPoint',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'Ham',
        'HandOutlines',
        'Herring',
        'HouseTwenty',
        'ItalyPowerDemand',
        'Lightning2',
        'MiddlePhalanxOutlineCorrect',
        'MoteStrain',
        'PhalangesOutlinesCorrect',
        'PowerCons',
        'ProximalPhalanxOutlineCorrect',
        'SemgHandGenderCh2',
        'ShapeletSim',
        'SonyAIBORobotSurface1',
        'SonyAIBORobotSurface2',
        'Strawberry',
        'ToeSegmentation1',
        'ToeSegmentation2',
        'TwoLeadECG',
        'Wafer',
        'Wine',
        'WormsTwoClass',
        'Yoga'
    ]

def get_three_classes_dataset_names():
    return [
        'ArrowHead',
        'CBF',
        'ChlorineConcentration',
        'DistalPhalanxOutlineAgeGroup',
        'LargeKitchenAppliances',
        'Meat',
        'MiddlePhalanxOutlineAgeGroup',
        'ProximalPhalanxOutlineAgeGroup',
        'RefrigerationDevices',
        'ScreenType',
        'SmallKitchenAppliances',
        'StarLightCurves',
        'BME',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'SmoothSubspace',
        'UMD'
    ]

def get_four_classes_dataset_names():
    return [
        'Car',
        'CinCECGTorso',
        'DiatomSizeReduction',
        'FaceFour',
        'OliveOil',
        'Trace',
        'TwoPatterns',
        'EthanolLevel',
        'Rock'
    ]

def get_multiclasses_dataset_names():
    return [
        'Beef', 'ECG5000', 'Haptics', 'Worms', 'MixedShapesRegularTrain', 'MixedShapesSmallTrain', 'SemgHandSubjectCh2', 'DistalPhalanxTW', 'MiddlePhalanxTW', 'OSULeaf', 'ProximalPhalanxTW', 'Symbols', 'SyntheticControl', 'GesturePebbleZ1', 'GesturePebbleZ2', 'SemgHandMovementCh2', 'ElectricDevices', 'Fish', 'InlineSkate', 'Lightning7', 'Plane', 'DodgerLoopDay', 'Mallat', 'UWaveGestureLibraryAll', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ', 'MedicalImages', 'ACSF1', 'AllGestureWiimoteX', 'AllGestureWiimoteY', 'AllGestureWiimoteZ', 'MelbournePedestrian', 'PickupGestureWiimoteZ', 'ShakeGestureWiimoteZ', 'InsectWingbeatSound', 'PLAID', 'CricketX', 'CricketY', 'CricketZ', 'EOGHorizontalSignal', 'EOGVerticalSignal', 'FaceAll', 'FacesUCR', 'SwedishLeaf', 'Fungi', 'Crop', 'WordSynonyms', 'GestureMidAirD1', 'GestureMidAirD2', 'GestureMidAirD3', 'Adiac', 'Phoneme', 'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2', 'FiftyWords', 'PigAirwayPressure', 'PigArtPressure', 'PigCVP', 'ShapesAll']


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
    return X_train, Y_train, X_test, Y_test

def preprocess_data(X_train, Y_train, X_test, Y_test):
    # TODO: Scale each time series?
    # TODO: NaN values? Should be none in the data
    # TODO: LabelEncoder necessary if we do not use PyTorch?
    return X_train, Y_train, X_test, Y_test

# Split data continuously
def split_data_continuous(features, target, num_clients):
    client_data = []
    data_per_client = len(features) // num_clients

    for i in range(num_clients):
        start_idx = i * data_per_client
        end_idx = (i + 1) * data_per_client
        client_features = features[start_idx:end_idx]
        client_target = target[start_idx:end_idx]
        client_data.append((client_features, client_target))

    return client_data

# Uniform split data with equal distribution
def split_data_uniform(features, target, num_clients, rng):
    # Check if number of clients is valid
    if num_clients <= 0:
        raise ValueError("Number of clients must be greater than 0.")
    unique_labels = np.unique(target)

    X_clients = [[] for _ in range(num_clients)]
    Y_clients = [[] for _ in range(num_clients)]
    
    shuffled_indices = rng.permutation(len(features))

    # Distribute one sample of each class to each client (if available)
    for class_label in unique_labels:
        class_indices = np.where(target == class_label)[0]
        rng.shuffle(class_indices)
        # Only distribute up to min(num_clients, number of samples for this class)
        num_to_distribute = min(num_clients, len(class_indices))
        for i in range(num_to_distribute):
            index = class_indices[i]
            X_clients[i].append(features[index])
            Y_clients[i].append(target[index])

    # Remaining samples in uniform distribution
    for index in shuffled_indices:
        x_sample, y_sample = features[index], target[index]
        # Skip if already distributed in the first phase
        already_distributed = any(
            np.array_equal(x_sample, x) and y_sample == y
            for client_x, client_y in zip(X_clients, Y_clients)
            for x, y in zip(client_x, client_y)
        )
        if not already_distributed:
            min_samples_client = min(range(num_clients), key=lambda k: len(Y_clients[k]))
            X_clients[min_samples_client].append(x_sample)
            Y_clients[min_samples_client].append(y_sample)

    # Convert lists to numpy arrays
    X_clients = [np.array(client) for client in X_clients]
    Y_clients = [np.array(client) for client in Y_clients]

    return X_clients, Y_clients

# Random split data
def split_data_random(features, target, num_clients):
    combined_df = pd.concat([features, target], axis=1)
    
    combined_df = combined_df.sample(frac=1, random_state=myseed).reset_index(drop=True)
    
    client_data = []
    data_per_client = len(combined_df) // num_clients

    for i in range(num_clients):
        start_idx = i * data_per_client
        end_idx = (i + 1) * data_per_client
        client_df = combined_df[start_idx:end_idx]
        client_features = client_df.iloc[:, :-1]
        client_target = client_df.iloc[:, -1]
        client_data.append((client_features, client_target))

    return client_data

def split_function(features, target, num_clients, random_state, my_split='uniform'):
    if my_split == 'continuous':
        return split_data_continuous(features, target, num_clients)
    elif my_split == 'uniform':
        return split_data_uniform(features, target, num_clients, random_state)
    elif my_split == 'random':
        return split_data_random(features, target, num_clients)
    elif my_split == 'dirichlet':  #train_features, train_targets, test_features, test_targets, num_clients, alpha=0.5, seed=None
        return split_data_dirichlet(features, target, num_clients, random_state)
    else:
        raise ValueError("Invalid split type. Choose 'continuous', 'uniform', or 'random'.")
