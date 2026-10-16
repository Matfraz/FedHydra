import numpy as np
from sklearn.metrics import (
    f1_score, 
    accuracy_score, 
    balanced_accuracy_score, 
    matthews_corrcoef, 
    cohen_kappa_score
)
import os
import random
import pandas as pd
from .centralized.run_hydra import hydra
from .centralized.run_quant import quant
from .centralized.run_minirocket import mini_rocket
from .centralized.run_rocket import rocket

from .utils.data_loader import get_binary_dataset_names, get_four_dataset_names, load_dataset, get_multiclass_dataset_name

def run_experiments():
    lists_of_dbs = get_binary_dataset_names() + get_four_dataset_names() + get_multiclass_dataset_name()

    models = {
        'Rocket': rocket,
        'MiniRocket': mini_rocket,
        'HydraClassifier': hydra,
        'QuantClassifier': quant
    }

    results = []

    for ds_name in lists_of_dbs:
        print('DATASET', ds_name)

        X_train, Y_train, X_test, Y_test = load_dataset(ds_name)
        is_multiclass = len(np.unique(Y_test)) > 2

        for model_name, model_func in models.items():
            print(f"Modello: {model_name} \n")
            Y_pred, test_time, train_time, peak_mem_train_mb = model_func( X_train, Y_train, X_test, Y_test)
            
            acc = accuracy_score(Y_test, Y_pred)
            b_acc = balanced_accuracy_score(Y_test, Y_pred)
            f1 = f1_score(Y_test, Y_pred, average='macro' if is_multiclass else 'binary')
            mcc = matthews_corrcoef(Y_test, Y_pred)
            kappa = cohen_kappa_score(Y_test, Y_pred)

            results.append({
                'Dataset': ds_name,
                'Algorithm': model_name,
                'Accuracy': acc,
                'F1_Score': f1,
                'Balanced_Accuracy': b_acc,
                'MCC': mcc,
                'Cohen_Kappa': kappa,
                'Train_Time_sec': train_time,
                'Test_Time_sec': test_time, 
                'Peak_RAM_Train_MB': peak_mem_train_mb
            })

    os.makedirs('results/centralized_metrics', exist_ok=True)
    df_results = pd.DataFrame(results)
    df_results.to_csv('results/centralized_metrics/all_baselines.csv', index=False)
    print("\nEsperimenti completati! Risultati salvati in all_baselines.csv")


if __name__ == '__main__':
    run_experiments()