import numpy as np
import pandas as pd
import random
import wandb
import argparse
import itertools
import os
import csv

from sklearn.linear_model import RidgeClassifier
from aeon.classification.convolution_based import HydraClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from utils import load_dataset, preprocess_data, split_function, RocketKernel, transform_seeds, get_binary_dataset_names, get_three_classes_dataset_names, get_four_classes_dataset_names, get_multiclasses_dataset_names, HydraGroup, hydra_transform_seeds
from sklearn.preprocessing import LabelBinarizer

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--clients', type=int, default=4, help='Choose the number of parties of the federation')
parser.add_argument('-k', '--kernels', type=int, default=8, help='Choose the number of HYDRA kernels to use')
parser.add_argument('-g', '--groups', type=int, default=8, help='Choose the number of HYDRA groups to use')
parser.add_argument('-r', '--rounds', type=int, default=100, help='Choose the number of rounds of federated training')
parser.add_argument('-v', '--validation', action='store_true', help='Choose this option to enable a grid search for the lambda regularizer on a validation set.')
parser.add_argument('--debug', action='store_true', help='Use this option to disable WandB metrics tracking')
args = parser.parse_args()

n_clients = args.clients 
n_kernels = args.kernels
n_rounds = args.rounds
n_groups = args.groups

# quanti dei migliori kernel ogni client potrà selezionare
top_k_per_client = (n_kernels * 1) // n_clients
EXPERIMENT_SEEDS = [1,2,3,4,5]

def hydra():
    #list_of_datasets =  get_binary_dataset_names() +  get_three_classes_dataset_names() + get_four_classes_dataset_names() + get_multiclasses_dataset_names()
    list_of_datasets = ['Car', 'Coffee','Computers', 'EOGHorizontalSignal']

    if args.debug:
            wandb.init(mode="disabled")
    else:
        None

    run_names = []
    # runs = wandb.Api().runs("mlgroup/Fed2RC")
    # for run in runs:
    #     run_names.append(run.name)
    # wandb.finish()
    
    for ds_name in list_of_datasets:
        print('DATASET', ds_name)

        X_train, Y_train, X_test, Y_test = load_dataset(ds_name)

        n_classes = len(np.unique(np.concatenate([Y_train, Y_test], axis=0)))
        X_train, Y_train, X_test, Y_test = preprocess_data(X_train, Y_train, X_test, Y_test)

        if args.debug:
            wandb.init(mode="disabled")
        else:
            entity = "mlgroup"
            project = "FedHydra"
            if args.validation:
                run_name = f"VAL_{ds_name}_{n_kernels}KERNELS_{n_clients}CLIENTS"
            else:
                run_name = f"{ds_name}_{n_kernels}KERNELS_{n_clients}CLIENTS"
            tags = ["FedHydra"]
            if run_name in run_names:
                print(f"Experiment {run_name} already executed.")
                continue
            else:
                wandb.init(project=project, entity=entity, group=f"{run_name}", name=run_name, tags=tags)

        for experiment_seed in EXPERIMENT_SEEDS:
            print('SEED', experiment_seed)
            # Questo garantisce che se riavvii lo script domani, la funzione "casuale" genererà l'esatta identica sequenza di numeri di oggi.
            rng = np.random.RandomState(experiment_seed)
            np.random.seed(experiment_seed)
            random.seed(experiment_seed)
            os.environ['PYTHONHASHSEED'] = str(experiment_seed)


            #partendo da un dataset unico (UCR Archive), il codice deve "frammentarlo" artificialmente e assegnarne un pezzo a ciascun finto client.
            if ds_name in ["DiatomSizeReduction", "Fungi", "PigAirwayPressure", "PigArtPressure", "PigCVP"]:
                split = "random"
                #Per questi cinque dataset specifici, lo sviluppatore ha deciso di usare uno split puramente casuale.
                s_X_Y_train = split_function(X_train, Y_train, n_clients, rng, my_split=split)
                s_X_Y_test = split_function(X_test, Y_test, n_clients, rng, my_split=split)
                s_X_train = [x for x, _ in s_X_Y_train]
                s_Y_train = [y for _, y in s_X_Y_train]
                s_X_test = [x for x, _ in s_X_Y_test]
                s_Y_test = [y for _, y in s_X_Y_test]
            else:
                #Per tutti gli altri dataset, viene forzata una divisione Uniforme. 
                # Questo significa che il programma fa molta attenzione a distribuire le classi 
                # in modo equo: se il dataset contiene cani, gatti e topi, si assicurerà 
                # (per quanto possibile) che ogni singolo client riceva nel suo pacchetto un po' di cani, un po' di gatti e un po' di topi.
                split='uniform'
                s_X_train, s_Y_train = split_function(X_train, Y_train, n_clients, rng, my_split=split)
                s_X_test, s_Y_test = split_function(X_test, Y_test, n_clients, rng, my_split=split)
            
            # Initialize different seeds for all clients
            # ogni client parte con un pacchetto di filtri iniziale unico.
            seeds = np.arange(n_clients * n_kernels).reshape(n_clients, n_kernels)

            dict_seeds = {}
            for client_id in range(n_clients):
                dict_seeds[client_id] = seeds[client_id]

            ts_length = len(X_train[0])
            n_classes = len(np.unique(Y_train))

            kernels = [[HydraGroup(seed=int(seed), ts_length=ts_length) for seed in dict_seeds[client_id]] for client_id in range(n_clients)]

            best_accuracy = 0
            best_seeds = None
            previous_seeds = 0
            consecutive_same_seeds = 0 
            selected_seeds = {}

            for epoch in range(n_rounds +1):
                train_accs = np.array([], dtype=float)
                round_selected_seeds = np.array([], dtype=int)
                train_accuracies = np.array([], dtype=float)
                lambdas = np.array([], dtype=float)

                for client_id in range(n_clients):
                    x_train, y_train = s_X_train[client_id], s_Y_train[client_id]
                    if args.validation:
                        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=experiment_seed)
                    # Il client legge quali seed (filtri) gli sono stati assegnati per questo round.
                    client_seeds = dict_seeds[client_id]
                    # Applica la trasformazione ROCKET (transform_seeds) per convertire le sue serie temporali in feature numeriche basate su quei filtr
                    x_trans = hydra_transform_seeds(x_train, client_seeds, ts_length)

                    model = RidgeClassifierCV(alphas=np.logspace(-3,0,100)).fit(x_trans, y_train)
                    # Il client estrae i "pesi" (w) che il suo modello appena addestrato ha assegnato a ciascun filtro.
                    w = model.coef_[0] 
                    train_acc = model.score(x_trans, y_train)
                    train_accuracies = np.concatenate([train_accuracies, [train_acc]])
                    
                    num_seeds = len(client_seeds)
                    features_per_group = x_trans.shape[1] // num_seeds

                    # Calcola quante feature produce ogni singolo HydraGroup
                    seed_scores = np.zeros(num_seeds)
                    for i in range(num_seeds):
                        start_idx = i * features_per_group
                        end_idx = start_idx + features_per_group

                        group_weights = w[start_idx:end_idx]
                        seed_scores[i] = np.sum(np.abs(group_weights))
                    
                    top_k_idx = np.argsort(-seed_scores)[:top_k_per_client]
                    top_k_seeds = client_seeds[top_k_idx]
                    round_selected_seeds = np.concatenate([round_selected_seeds, top_k_seeds])
                    lambdas = np.concatenate([lambdas, [model.alpha_ * train_acc]])
                    train_accs = np.concatenate([train_accs, [train_acc]])

                #l server virtuale prende l'urna globale piena dei seed vincenti proposti da tutti i nodi e usa np.unique per eliminare i doppion
                round_selected_seeds = np.unique(round_selected_seeds)
                # alcola quanti kernel univoci sono sopravvissuti in questo round
                final_num_kernels = len(round_selected_seeds)
                # Calcola e salva la media delle accuratezze riportate dai client per avere un'idea di come sta andando l'addestramento globale.
                mean_train_accuracy = np.mean(train_accuracies)


                #monitoraggio, l'interruzione intelligente (early stopping) e la ridistribuzione della conoscenza.
                print(f"\n=== Round {epoch + 1} ===, Mean Train Accuracy: {mean_train_accuracy:.4f}, Number of unique seeds: {final_num_kernels}")

                if previous_seeds is not None and np.array_equal(previous_seeds, round_selected_seeds):
                    consecutive_same_seeds += 1
                    if consecutive_same_seeds >= 1: 
                        print("\n Convergence achieved: Seeds unchanged for 2 consecutive epochs. Training terminated.")
                        break
                else:
                    consecutive_same_seeds = 0
                
                previous_seeds = round_selected_seeds.copy()

                #sostituisce il loro dizionario personale di filtri (dict_seeds[client_id]) con la nuova lista dei filtri d'élite aggregati dal server
                for client_id in range(n_clients):
                    dict_seeds[client_id] = round_selected_seeds.copy()

            A_global = None
            b_global = None
            A_locals, b_locals = [], []
            for client_id in range(n_clients):
                x_train, y_train = s_X_train[client_id], s_Y_train[client_id]
                # ora questi seed non sono più casuali, ma sono la lista finale dei kernel migliori condivisa da tutti
                client_seeds = dict_seeds[client_id]
            
                x_trans = hydra_transform_seeds(x_train, client_seeds, ts_length)  

                # Converte le etichette (es. la classe 2) in formato "One-Hot" (es. [0, 0, 1]
                y_onehot = np.eye(n_classes)[y_train]

                # Impossibile fare il processo inverso
                A_local = x_trans.T @ x_trans
                b_local = x_trans.T @ y_onehot

                # Se è il primo client, capiamo la dimensione REALE delle feature e creiamo le matrici globali
                if A_global is None:
                    n_features_reali = A_local.shape[0] 
                    A_global = np.zeros((n_features_reali, n_features_reali))
                    b_global = np.zeros((n_features_reali, n_classes))

                A_locals.append(A_local)
                b_locals.append(b_local)
                        
                A_global += A_local
                b_global += b_local


            #cross validation for best lambda
            local_best_lambdas = np.array([])
            local_best_accs = np.array([])
            tmp_lambda = 0
                                        