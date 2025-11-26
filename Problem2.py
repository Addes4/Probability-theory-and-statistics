## Problem 2: Maximum likelihood, minsta kvadrat

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import Data_and_tools as tools

M = 100000
b = 4
# Simulera M utfall med parameter b.
x = stats.rayleigh.rvs(scale=b, size=M)
# Skapa figur och plotta histogrammet.
plt.figure()
plt.hist(x, 40, density=True)
est_ml = np.sqrt(np.mean(x**2) / 2) # Skriv din ML-skattning här (ej rätt just nu)
est_mk = np.sqrt(2/np.pi) * np.mean(x) # Skriv din MK-skattning här (ej rätt just nu)
# Plotta de två skattningarna.
plt.plot(est_ml, 0.2, 'r*', markersize=10)
plt.plot(est_mk, 0.2, 'g*', markersize=10)
plt.plot(b, 0.2, 'bo')
plt.show()


## Problem 2: Maximum likelihood, minsta kvadrat (forts.)
# Skapa figur.
plt.figure()
# Visa histogrammet.
plt.hist(x, 40, density=True)
# Plotta täthetsfunktionen för den skattade parametern.
x_grid = np.linspace(np.min(x), np.max(x), 60)
pdf = stats.rayleigh.pdf(x_grid, scale=est_ml)
plt.plot(x_grid, pdf, 'r')
plt.show()