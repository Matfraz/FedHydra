import numpy as np
import pandas as pd
import random
import wandb
import argparse
import itertools
import os
import csv

from sklearn.linear_model import RidgeClassifier
from sktime.transformations.panel.rocket import Rocket
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from utils import load_dataset, preprocess_data, split_function, RocketKernel, transform_seeds, get_binary_dataset_names, get_three_classes_dataset_names, get_four_classes_dataset_names, get_multiclasses_dataset_names
from sklearn.preprocessing import LabelBinarizer

# Lettura da riga di comando
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--clients', type=int, default=4, help='Choose the number of parties of the federation')
parser.add_argument('-k', '--kernels', type=int, default=100, help='Choose the number of ROCKET kernels to use')
parser.add_argument('-r', '--rounds', type=int, default=100, help='Choose the number of rounds of federated training')
parser.add_argument('-v', '--validation', action='store_true', help='Choose this option to enable a grid search for the lambda regularizer on a validation set.')
parser.add_argument('--debug', action='store_true', help='Use this option to disable WandB metrics tracking')
args = parser.parse_args()

# Constants
n_clients = args.clients 
n_kernels = args.kernels
n_rounds = args.rounds

# quanti dei migliori kernel ogni client potrà selezionare
top_k_per_client = (n_kernels * 1) // n_clients
EXPERIMENT_SEEDS = [1,2,3,4,5]

def main():
    #list_of_datasets = get_binary_dataset_names()
    #list_of_datasets = get_three_classes_dataset_names()
    #list_of_datasets = get_four_classes_dataset_names()
    #list_of_datasets = get_multiclasses_dataset_names()
    #list_of_datasets = ['PigCVP'] #example
    # Costruisce un'unica grande lista contenente i nomi di tutti i 
    # dataset disponibili nell'archivio UCR.
    list_of_datasets = ['Car', 'Coffee','Computers', 'EOGHorizontalSignal']

    #Se è vera, dice esplicitamente alla libreria wandb di disattivarsi (mode="disabled"). 
    # In questo modo puoi far girare il codice sul tuo computer 
    # per cercare errori, senza che invii grafici o dati al server cloud del tuo team.
    if args.debug:
        wandb.init(mode="disabled")
    else:
        None

    # All the runs
    # Se il dataset corrente è già presente in run_names, saltalo e passa al prossimo, 
    # perché lo abbiamo già calcolato nei giorni scorsi
    run_names = []
    # runs = wandb.Api().runs("mlgroup/Fed2RC")
    # for run in runs:
    #     run_names.append(run.name)
    # wandb.finish()

    # Inizia un ciclo for che itererà su ogni dataset 
    # Per ciascuno, usa la funzione load_dataset per caricare le variabili: 
    # le X sono le features (i valori numerici delle serie temporali) e le Y labels.
    for ds_name in list_of_datasets:
        print('DATASET', ds_name)

        X_train, Y_train, X_test, Y_test = load_dataset(ds_name)

        # Prima calcola il numero esatto di classi (n_classes) previste da questo dataset. 
        # Lo fa concatenando le etichette di addestramento (Y_train) e 
        # quelle di test (Y_test) per essere sicuro di non perdersi nessuna classe 
        #(nel caso raro in cui una classe sia presente solo nel set di test), 
        # dopodiché conta quanti valori unici ci sono con np.unique
        n_classes = len(np.unique(np.concatenate([Y_train, Y_test], axis=0)))
        X_train, Y_train, X_test, Y_test = preprocess_data(X_train, Y_train, X_test, Y_test)

        #Se non siamo in modalità debug, il codice costruisce dinamicamente il nome dell'esperimento (run_name).
        if args.debug:
            wandb.init(mode="disabled")
        else:
            entity = "mlgroup"
            project = "Fed2RC"
            if args.validation:
                run_name = f"VAL_{ds_name}_{n_kernels}KERNELS_{n_clients}CLIENTS"
            else:
                run_name = f"{ds_name}_{n_kernels}KERNELS_{n_clients}CLIENTS"
            tags = ["Fed2RC"]
            if run_name in run_names:
                print(f"Experiment {run_name} already executed.")
                continue
            else:
                wandb.init(project=project, entity=entity, group=f"{run_name}", name=run_name, tags=tags)

        # l codice avvia un ciclo per ripetere l'intero addestramento più volte usando i seed 1, 2, 3, 4, 5 definiti in precedenza.
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

            # crea un dizionario Python per associare in modo ordinato e 
            # facilmente recuperabile ogni ID del client (es. 0, 1, 2, 3) 
            # alla sua specifica riga della matrice (la sua lista personale di 100 numeri interi).
            dict_seeds = {}
            for client_id in range(n_clients):
                dict_seeds[client_id] = seeds[client_id]

            #Lettura delle dimensioni dei dati
            ts_length = len(X_train[0])
            n_classes = len(np.unique(Y_train))
            #Inizializzazione fisica dei Kernel ROCKET
            #kernels = [[RocketKernel(seed=int(seed), ts_length=ts_length, ppv_only=True) for seed in dict_seeds[client_id]] for client_id in range(n_clients)]

            #variabili di tracciamento e controllo.
            best_accuracy = 0
            best_seeds = None
            previous_seeds = 0
            consecutive_same_seeds = 0 
            selected_seeds = {}

            for epoch in range(n_rounds +1):
                # All'inizio di ogni nuovo round, "azzera" delle liste vuote 
                # (usando array di NumPy) che serviranno al server per raccogliere i risultati 
                #che i client gli manderanno (accuratezza, lambda e, soprattutto, i 
                #round_selected_seeds, ovvero l'urna globale dove finiranno i kernel proposti da tutti).
                train_accs = np.array([], dtype=float)
                round_selected_seeds = np.array([], dtype=int)
                train_accuracies = np.array([], dtype=float)
                lambdas = np.array([], dtype=float)

                for client_id in range(n_clients):
                    #Il ciclo itera su ciascun nodo uno alla volta . Il client preleva la sua porzione di dati privati.
                    x_train, y_train = s_X_train[client_id], s_Y_train[client_id]
                    if args.validation:
                        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=experiment_seed)
                    # Il client legge quali seed (filtri) gli sono stati assegnati per questo round.
                    client_seeds = dict_seeds[client_id]
                    # Applica la trasformazione ROCKET (transform_seeds) per convertire le sue serie temporali in feature numeriche basate su quei filtr
                    x_trans = transform_seeds(x_train, client_seeds, ts_length)

                    # Addestra un classificatore locale (RidgeClassifierCV) sui suoi dati trasformati, calcolando in automatico (grazie al CV, Cross Validation) il miglior parametro di regolarizzazione per non imparare i dati a memoria.
                    model = RidgeClassifierCV(alphas=np.logspace(-3,0,100)).fit(x_trans, y_train)
                    # Il client estrae i "pesi" (w) che il suo modello appena addestrato ha assegnato a ciascun filtro.
                    w = model.coef_[0] 
                    train_acc = model.score(x_trans, y_train)
                    train_accuracies = np.concatenate([train_accuracies, [train_acc]])
                    top_k_idx = np.argsort(-np.abs(w))[:top_k_per_client]
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

                # La funzione np.array_equal confronta l'array contenente i kernel vincenti di questo round (round_selected_seeds) 
                # con quelli del round precedente (previous_seeds).
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
                        

            # Questo blocco di codice viene eseguito dopo che il ciclo dei round si è concluso (cioè dopo che i client hanno raggiunto la "convergenza" e si sono accordati sui kernel migliori).
            # la costruzione del modello globale senza mai condividere i dati originali.
            # prepara le due matrici vuote (A_global e b_global) che conterranno la conoscenza di tutta la rete. 
            # La loro dimensione si basa sul numero finale di kernel vincenti (final_num_kernels).
            A_global = np.zeros((final_num_kernels, final_num_kernels))
            b_global = np.zeros((final_num_kernels, n_classes))
            A_locals, b_locals = [], []
            for client_id in range(n_clients):
                x_train, y_train = s_X_train[client_id], s_Y_train[client_id]
                # ora questi seed non sono più casuali, ma sono la lista finale dei kernel migliori condivisa da tutti
                client_seeds = dict_seeds[client_id]
            
                x_trans = transform_seeds(x_train, client_seeds, ts_length)  

                # Converte le etichette (es. la classe 2) in formato "One-Hot" (es. [0, 0, 1]
                y_onehot = np.eye(n_classes)[y_train]

                # Impossibile fare il processo inverso
                A_local = x_trans.T @ x_trans
                b_local = x_trans.T @ y_onehot
                A_locals.append(A_local)
                b_locals.append(b_local)
                        
                A_global += A_local
                b_global += b_local


            #cross validation for best lambda
            local_best_lambdas = np.array([])
            local_best_accs = np.array([])
            tmp_lambda = 0
            for client_id in range(n_clients):
                best_accuracy = 0
                x_train, y_train = s_X_train[client_id], s_Y_train[client_id]

                # Se dal terminale è stata attivata la modalità validazione (args.validation), il client prende i suoi dati di addestramento e ne separa un 10% per usarlo come test interno
                if args.validation:
                    _, x_val, _, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=experiment_seed)
                client_seeds = dict_seeds[client_id]
                x_trans = transform_seeds(x_train, client_seeds, ts_length)  
                if args.validation:
                    x_trans = transform_seeds(x_val, client_seeds, ts_length)
                y_onehot = np.eye(n_classes)[y_train]

                #  Il client prende la matrice globale (che contiene la conoscenza di tutti) e sottrae la propria matrice locale.   
                # Questo serve per verificare se il modello è in grado di generalizzare su dati che non ha mai visto
                A_minus_client = A_global - A_locals[client_id]
                b_minus_client = b_global - b_locals[client_id]

                # Inizia un ciclo che testa 100 valori diversi di $\lambda$. 
                for cv_lambda in np.logspace(-3,0,100):
                    if args.validation == False:
                        # Questa è l'implementazione algebrica della regolarizzazione Ridge.
                        local_A_final = A_minus_client + cv_lambda * np.eye(A_global.shape[0])
                        # Usa np.linalg.solve per risolvere il sistema lineare ($AW = b$) e calcolare i pesi finali del modello (local_W_final).
                        local_W_final = np.linalg.solve(local_A_final, b_minus_client)
                    else:
                        local_A_final = A_global + cv_lambda * np.eye(A_global.shape[0])
                        local_W_final = np.linalg.solve(local_A_final, b_global)
                    local_W_final =  local_W_final / (np.linalg.norm(local_W_final, axis=0, keepdims=True) + 1e-10)
                    local_y_pred = np.argmax(x_trans @ local_W_final, axis=1)
                    if args.validation == False:
                        local_accuracy = accuracy_score(y_train, local_y_pred)
                    else:
                        local_accuracy = accuracy_score(y_val, local_y_pred)
                    
                    # Se l'accuratezza ottenuta con questo specifico $\lambda$ è migliore di quella registrata fino a quel momento, aggiorna le variabili salvando la nuova best_accuracy e il tmp_lambda vincente. 
                    if best_accuracy < local_accuracy or (best_accuracy == local_accuracy and cv_lambda > tmp_lambda):
                        tmp_lambda = cv_lambda
                        best_accuracy = local_accuracy
                    #print("Client: ", client_id, "best local accuracy: ", local_accuracy, "with lambda: ", cv_lambda)
                local_best_lambdas = np.concatenate([local_best_lambdas, [tmp_lambda * best_accuracy]])
                local_best_accs = np.concatenate([local_best_accs, [best_accuracy]])

            A_final = A_global + np.sum(local_best_lambdas)/np.sum(local_best_accs) * np.eye(A_global.shape[0])
            W_final = np.linalg.solve(A_final, b_global)
            W_final =  W_final / (np.linalg.norm(W_final, axis=0, keepdims=True) + 1e-10)
            X_test_trans = transform_seeds(X_test, round_selected_seeds, ts_length)
            y_pred = np.argmax(X_test_trans @ W_final, axis=1)
            test_accuracy = accuracy_score(Y_test, y_pred)
            if n_classes == 2:
                test_f1 = f1_score(Y_test, y_pred)
            else:
                test_f1 = f1_score(Y_test, y_pred, average = 'macro')

            print(f"Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1}, Number of unique seeds: {n_kernels}")

            if args.validation:    
                wandb.log({f'Test Accuracy using a regularizer coming from a validation set ({experiment_seed})': test_accuracy,
                            f'Test F1 using a regularizer coming from a validation set ({experiment_seed})': test_f1,
                            f'Validation Regularizer': np.sum(local_best_lambdas)/np.sum(local_best_accs),
                            f'Rounds': ({epoch}),
                            f'Kernels': final_num_kernels})
            else:
                wandb.log({f'Test Accuracy ({experiment_seed})': test_accuracy,
                            f'Test F1 ({experiment_seed})': test_f1,
                            f'Validation Regularizer': np.sum(local_best_lambdas)/np.sum(local_best_accs),
                            f'Rounds': ({epoch}),
                            f'Kernels': final_num_kernels})
            csv_path = 'results/federated_metrics/fed2rc_results_5seeds_10kkernel_2.csv'
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            file_exists = os.path.isfile(csv_path)
            
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Dataset", "Seed", "Test_Accuracy", "Test_F1", "Rounds_to_Converge", "Final_Kernels"])
                writer.writerow([ds_name, experiment_seed, test_accuracy, test_f1, epoch, final_num_kernels])
        wandb.finish()

if __name__ == '__main__':
    main()