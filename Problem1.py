## Problem 1: Simulering av konfidensintervall
# Parametrar
# Antal mätningar

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import Data_and_tools as tools

N = 25
# Väntevärdet
MU = 2
# Standardavvikelsen
SIGMA = 1
# Ett minus konfidensgraden
ALPHA = 0.05
# Antal intervall
M = 100
# Simulera n observationer för varje intervall.
x = stats.norm.rvs(loc=MU, scale=SIGMA, size=(M, N))
# Skatta mu med medelvärdet.
xbar = np.mean(x, axis=-1)
# Beräkna kvantilerna och standardavvikelsen för
# medelvärdet.
lambda_alpha_2 = stats.norm.ppf(1 - ALPHA / 2)
D = SIGMA / np.sqrt(N)
# Beräkna undre och övre gränserna.
undre = xbar - lambda_alpha_2 * D
övre = xbar + lambda_alpha_2 * D

## Problem 1: Simulering av konfidensintervall (forts.)
# Skapa en figur med storlek 4 Ö 8 tum.
plt.figure(figsize=(4, 8))
# Rita upp alla intervall
for k in range(M):
    # Rödmarkera alla intervall som missar mu.
    if övre[k] < MU or undre[k] > MU:
        color = 'r'
    else:
        color = 'b'
    plt.plot([undre[k], övre[k]], [k, k], color)

# Fixa till gränserna så att figuren ser lite bättre ut.
b_min = np.min(undre)

b_max = np.max(övre)
plt.axis([b_min, b_max, -1, M])
# Rita ut det sanna värdet.
plt.plot([MU, MU], [-1, M], 'g')
# Visa plotten.
plt.show()