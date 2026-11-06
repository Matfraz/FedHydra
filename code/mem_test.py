import sys, time, resource
import numpy as np
from utils import hydra_transform_seeds

g, k, n, L = map(int, sys.argv[1:5])   # gruppi, kernel, n. serie, lunghezza serie
X = np.random.randn(n, L)

t = time.time()
F = hydra_transform_seeds(X, range(g), k, L)
picco = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024   # KB -> MB su Linux
print(f"feature {F.shape} | {time.time()-t:.1f} s | picco RAM {picco:.0f} MB")