import numpy as np
import pandas as pd
import random
import wandb
import argparse
import itertools
import os
import csv
import torch
import time

from sklearn.linear_model import RidgeClassifier
from aeon.transformations.collection.convolution_based._hydra import HydraTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from utils import load_dataset, preprocess_data, split_function, RocketKernel,transform_seeds, get_binary_dataset_names, get_three_classes_dataset_names, get_four_classes_dataset_names, get_multiclasses_dataset_names, HydraGroup, hydra_transform_seeds, SparseScaler, split_data_dirichlet
from sklearn.preprocessing import LabelBinarizer

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--clients', type=int, default=4, help='Choose the number of parties of the federation')
parser.add_argument('-k', '--kernels', type=int, default=8, help='Choose the number of HYDRA kernels to use')
parser.add_argument('-g', '--groups', type=int, default=64, help='Choose the number of HYDRA groups to use')
parser.add_argument('-r', '--rounds', type=int, default=100, help='Choose the number of rounds of federated training')
parser.add_argument('--split', choices=['iid', 'dirichlet'], default='iid')
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--file', type=str, default="fedhydra.csv", help='Choose the name of the file')
parser.add_argument('-v', '--validation', action='store_true', help='Choose this option to enable a grid search for the lambda regularizer on a validation set.')
parser.add_argument('--debug', action='store_true', help='Use this option to disable WandB metrics tracking')
args = parser.parse_args()

n_clients = args.clients 
n_kernels = args.kernels
n_rounds = args.rounds
n_groups = args.groups

file_name = args.file

top_k_per_client = max(1, (n_groups * 1) // n_clients)
EXPERIMENT_SEEDS = [1,2,3,4,5]

def hydra():
    #list_of_datasets =  get_binary_dataset_names() +  get_three_classes_dataset_names() + get_four_classes_dataset_names() + get_multiclasses_dataset_names()
    list_of_datasets = ['SemgHandSubjectCh2','Car', 'Coffee','Computers', 'EOGHorizontalSignal', 'Adiac']

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
            entity = "matfraz-universit-di-torino"
            project = "FedHydra"
            split_tag = "IID" if args.split == 'iid' else f"DIR{args.alpha}"
            if args.validation:
                run_name = f"VAL_{ds_name}_{n_groups}GROUPS_{n_clients}CLIENTS_{split_tag}"
            else:
                run_name = f"{ds_name}_{n_groups}GROUPS_{n_clients}CLIENTS_{split_tag}"
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
            if args.split == 'dirichlet':
                try:
                    split = 'dirichlet'
                    s_X_train, s_Y_train, s_X_test, s_Y_test = split_data_dirichlet( X_train, Y_train, X_test, Y_test, n_clients,alpha=args.alpha, seed=experiment_seed)
                except Exception as e:
                    print(f"{ds_name}: Dirichlet non possibile ({e}) -> uso random")
                    tr = split_function(X_train, Y_train, n_clients, rng, my_split='random')
                    te = split_function(X_test,  Y_test,  n_clients, rng, my_split='random')
                    s_X_train = [x for x, _ in tr]; s_Y_train = [y for _, y in tr]
                    s_X_test  = [x for x, _ in te]; s_Y_test  = [y for _, y in te]
                    split = 'random'
            elif ds_name in ["DiatomSizeReduction", "Fungi", "PigAirwayPressure", "PigArtPressure", "PigCVP"]:
                split = "random"
                s_X_Y_train = split_function(X_train, Y_train, n_clients, rng, my_split=split)
                s_X_Y_test = split_function(X_test, Y_test, n_clients, rng, my_split=split)
                s_X_train = [x for x, _ in s_X_Y_train]
                s_Y_train = [y for _, y in s_X_Y_train]
                s_X_test = [x for x, _ in s_X_Y_test]
                s_Y_test = [y for _, y in s_X_Y_test]
            else:
                split='uniform'
                s_X_train, s_Y_train = split_function(X_train, Y_train, n_clients, rng, my_split=split)
                s_X_test, s_Y_test = split_function(X_test, Y_test, n_clients, rng, my_split=split)

            # Initialize different seeds for all clients
            # ogni client parte con un pacchetto di filtri iniziale unico.
            seeds = np.arange(n_clients * n_groups).reshape(n_clients, n_groups)

            dict_seeds = {}
            for client_id in range(n_clients):
                dict_seeds[client_id] = seeds[client_id]

            ts_length = len(X_train[0])
            n_classes = len(np.unique(Y_train))

            # kernels = [[HydraGroup(seed=int(seed), ts_length=ts_length) for seed in dict_seeds[client_id]] for client_id in range(n_clients)]

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
                    # Applica la trasformazione Hydra (transform_seeds) per convertire le sue serie temporali in feature numeriche basate su quei filtr
                    x_trans = hydra_transform_seeds(x_train, client_seeds, n_kernels,ts_length)

                    # Normalizzazione tramite SparseScaler
                    x_trans_tensor = torch.tensor(x_trans, dtype=torch.float32)
                    scaler = SparseScaler()
                    x_trans_scaled_tensor = scaler.fit_transform(x_trans_tensor)

                    x_trans_normalized = x_trans_scaled_tensor.numpy()

                    model = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).fit(x_trans_normalized, y_train)
                    
                    w = np.sum(np.abs(model.coef_), axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
                    train_acc = model.score(x_trans_normalized, y_train)
                    train_accuracies = np.concatenate([train_accuracies, [train_acc]])
                    
                    num_seeds = len(client_seeds)
                    features_per_group = x_trans_normalized.shape[1] // num_seeds

                    # Calcola quante feature produce ogni singolo HydraGroup
                    seed_scores = np.zeros(num_seeds)
                    for i in range(num_seeds):
                        start_idx = i * features_per_group
                        end_idx = start_idx + features_per_group

                        group_weights = w[start_idx:end_idx]
                        seed_scores[i] = np.sum(np.abs(group_weights))
                        #seed_scores[i] = np.max(np.abs(group_weights))
                        #seed_scores[i] = np.sum(np.sort(np.abs(group_weights))[-5:])
                    
                    top_k_idx = np.argsort(-seed_scores)[:top_k_per_client]
                    top_k_seeds = client_seeds[top_k_idx]
                    round_selected_seeds = np.concatenate([round_selected_seeds, top_k_seeds])
                    lambdas = np.concatenate([lambdas, [model.alpha_ * train_acc]])
                    train_accs = np.concatenate([train_accs, [train_acc]])

                #l server virtuale prende l'urna globale piena dei seed vincenti proposti da tutti i nodi e usa np.unique per eliminare i doppion
                round_selected_seeds = np.unique(round_selected_seeds)
               
                final_num_groups = len(round_selected_seeds)
                # Calcola e salva la media delle accuratezze riportate dai client per avere un'idea di come sta andando l'addestramento globale.
                mean_train_accuracy = np.mean(train_accuracies)


                #monitoraggio, l'interruzione intelligente (early stopping) e la ridistribuzione della conoscenza.
                print(f"\n=== Round {epoch + 1} ===, Mean Train Accuracy: {mean_train_accuracy:.4f}, Number of unique seeds: {final_num_groups}")

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

            
            list_x_trans = []
            sum_s = []
            sum_s_squared = []
            sum_zeros = []
            N_samples = []
            for client_id in range(n_clients):
                x_train, y_train = s_X_train[client_id], s_Y_train[client_id]
                client_seeds = dict_seeds[client_id]
 
                x_trans = hydra_transform_seeds(x_train, client_seeds, n_kernels, ts_length)
                list_x_trans.append(x_trans)

                s = np.sqrt(np.maximum(np.asarray(x_trans, dtype=np.float64), 0.0))   # = X.clamp(0).sqrt()
                sum_s.append(np.sum(s, axis=0))
                sum_s_squared.append(np.sum(np.square(s), axis=0))
                sum_zeros.append(np.sum(s == 0, axis=0))
                N_samples.append(len(s))

            
            N_samples_global = np.sum(N_samples)
            sum_s_global = np.sum(sum_s, axis=0)
            sum_s_squared_global = np.sum(sum_s_squared, axis=0)
            sum_zeros_global = np.sum(sum_zeros, axis=0)
 
            mu_global = sum_s_global / N_samples_global
            # varianza campionaria (denominatore n-1, come torch.std); np.maximum evita negativi da arrotondamento
            varianza_global = np.maximum(0.0, (sum_s_squared_global - N_samples_global * np.square(mu_global)) / (N_samples_global - 1))
            epsilon_global = (sum_zeros_global / N_samples_global) ** 4 + 1e-8
            sigma_global = np.sqrt(varianza_global) + epsilon_global
            # mu_global, sigma_global vengono rimandati a tutti i client

            A_global = None
            b_global = None
            A_locals, b_locals, list_x_scaled = [], [], []

            for client_id in range(n_clients):
                y_train = s_Y_train[client_id]
 
                # Stessa trasformazione dello SparseScaler (mask=True), con i parametri globali
                s = np.sqrt(np.maximum(np.asarray(list_x_trans[client_id], dtype=np.float64), 0.0))
                x_trans_scaled = ((s - mu_global) * (s != 0)) / sigma_global
                list_x_scaled.append(x_trans_scaled)
                # Converte le etichette (es. la classe 2) in formato "One-Hot" (es. [0, 0, 1]
                y_onehot = np.eye(n_classes)[y_train]
                # Impossibile fare il processo inverso
                A_local = x_trans_scaled.T @ x_trans_scaled  # Matrice di covarianza delle feature
                b_local = x_trans_scaled.T @ y_onehot # Correlazione tra feature e classi (Z^T * Y)

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
            

            for client_id in range(n_clients):
                tmp_lambda = 0
                best_accuracy = 0
                x_train, y_train = s_X_train[client_id], s_Y_train[client_id]

                if args.validation:
                    _, x_val, _, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=experiment_seed)
                    client_seeds = dict_seeds[client_id]
                    x_val_trans = hydra_transform_seeds(x_val, client_seeds, n_kernels, ts_length)
                    s = np.sqrt(np.maximum(np.asarray(x_val_trans, dtype=np.float64), 0.0))
                    x_trans = ((s - mu_global) * (s != 0)) / sigma_global
                    target_A = A_global
                    target_b = b_global
                else:
                    # riusa le feature già trasformate e scalate, niente nuova trasformazione
                    x_trans = list_x_scaled[client_id]
                    target_A = A_global - A_locals[client_id]
                    target_b = b_global - b_locals[client_id]

                A_values, A_vectors = np.linalg.eigh(target_A)

                c = A_vectors.T @ target_b

                for cv_lambda in np.logspace(-3, 3, 10):
                    reg_eigenvalues = A_values[:, np.newaxis] + cv_lambda
                    spectral_weights = c / reg_eigenvalues
                    local_W_final = A_vectors @ spectral_weights
                    
                    local_W_final =  local_W_final / (np.linalg.norm(local_W_final, axis=0, keepdims=True) + 1e-10)
                    local_y_pred = np.argmax(x_trans @ local_W_final, axis=1)
                    if args.validation == False:
                        local_accuracy = accuracy_score(y_train, local_y_pred)
                    else:
                        local_accuracy = accuracy_score(y_val, local_y_pred)

                    if best_accuracy < local_accuracy or (best_accuracy == local_accuracy and cv_lambda > tmp_lambda):
                        tmp_lambda = cv_lambda
                        best_accuracy = local_accuracy
                    #print("Client: ", client_id, "best local accuracy: ", local_accuracy, "with lambda: ", cv_lambda)
                local_best_lambdas = np.concatenate([local_best_lambdas, [tmp_lambda * best_accuracy]])
                local_best_accs = np.concatenate([local_best_accs, [best_accuracy]])

            total_acc = np.sum(local_best_accs)
            if total_acc > 0:
                lam = np.sum(local_best_lambdas) / total_acc
            else:
                lam = 1.0   
            lam = max(lam, 1e-3)   
            #print(lam)

            A_final = A_global + lam * np.eye(A_global.shape[0])
            W_final = np.linalg.solve(A_final, b_global)
            W_final =  W_final / (np.linalg.norm(W_final, axis=0, keepdims=True) + 1e-10)
            X_test_trans = hydra_transform_seeds(X_test, round_selected_seeds, n_kernels, ts_length)
            s = np.sqrt(np.maximum(np.asarray(X_test_trans, dtype=np.float64), 0.0))
            X_test_trans = ((s - mu_global) * (s != 0)) / sigma_global
            y_pred = np.argmax(X_test_trans @ W_final, axis=1)
            test_accuracy = accuracy_score(Y_test, y_pred)
            if n_classes == 2:
                test_f1 = f1_score(Y_test, y_pred)
            else:
                test_f1 = f1_score(Y_test, y_pred, average = 'macro')

            print(f"Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1}, Number of unique seeds: {final_num_groups}")

            if args.validation:    
                wandb.log({f'Test Accuracy using a regularizer coming from a validation set ({experiment_seed})': test_accuracy,
                            f'Test F1 using a regularizer coming from a validation set ({experiment_seed})': test_f1,
                            f'Validation Regularizer': lam,
                            f'Rounds': ({epoch}),
                            f'Kernels': final_num_groups})
            else:
                wandb.log({f'Test Accuracy ({experiment_seed})': test_accuracy,
                            f'Test F1 ({experiment_seed})': test_f1,
                            f'Validation Regularizer': lam,
                            f'Rounds': ({epoch}),
                            f'Kernels': final_num_groups})
            csv_path = f"results/{file_name}"
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            file_exists = os.path.isfile(csv_path)
            
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Dataset", "Seed", "Test_Accuracy", "Test_F1", "Rounds_to_Converge", "Final_Groups"])
                writer.writerow([ds_name, experiment_seed, test_accuracy, test_f1, epoch, final_num_groups])
        wandb.finish()
            
if __name__ == '__main__':
    t0 = time.perf_counter()
    hydra()
    elapsed = time.perf_counter() - t0
    print(f"\nTempo totale: {elapsed:.1f} s ({elapsed / 60:.1f} min)")

    tempo_str = f"{file_name} = {elapsed:.1f} s ({elapsed / 60:.1f} min)\n"
    os.makedirs('results', exist_ok=True)
    with open('results/tempo_totale.txt', mode='a') as f:
        f.write(tempo_str)
