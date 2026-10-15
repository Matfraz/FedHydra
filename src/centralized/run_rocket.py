from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import f1_score, classification_report

import numpy as np
import time
import os
import random
import tracemalloc


def rocket(X_train, Y_train, X_test, Y_test):
    rocket = Rocket(num_kernels = 10000, random_state = 42)

    tracemalloc.start()
    start_train = time.time()
    
    rocket.fit(X_train)
    X_train_transform = rocket.transform(X_train)
    X_test_transform = rocket.transform(X_test)

    train_time = time.time() - start_train
    _, peak_mem_train = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_train_mb = peak_mem_train / (1024 * 1024)
    
    alphas = np.logspace(-3, 0, 10)
    classifier = RidgeClassifierCV(alphas=alphas)
    classifier.fit(X_train_transform, Y_train)

    start_test = time.time()
    Y_pred = classifier.predict(X_test_transform)
    test_time = time.time() - start_test

    return Y_pred,test_time,train_time, peak_mem_train_mb


if __name__ == '__main__':
    rocket()

            
