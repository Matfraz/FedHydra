from aeon.classification.interval_based import QUANTClassifier
from sklearn.metrics import f1_score, classification_report

import numpy as np
import os
import random
import time
import tracemalloc

def quant(X_train, Y_train, X_test, Y_test):
    quant = QUANTClassifier(random_state = 42)

    tracemalloc.start()
    start_train = time.time()
    quant.fit(X_train,Y_train)
    train_time = time.time() - start_train

    _, peak_mem_train = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_train_mb = peak_mem_train / (1024 * 1024)
    
    start_test = time.time()
    Y_pred = quant.predict(X_test)
    test_time = time.time() - start_test
    
    return Y_pred,test_time,train_time, peak_mem_train_mb  


if __name__ == '__main__':
    quant()

            