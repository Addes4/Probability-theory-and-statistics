## Problem 3: Konfidensintervall for Rayleighfordelning

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import Data_and_tools as tools

# Ladda data.
y = np.loadtxt('wave_data.dat')
# Plotta en bit av signalen samt histogrammet.
plt.figure(figsize=(4, 8))
plt.subplot(2, 1, 1)
plt.plot(y[:100])
plt.subplot(2, 1, 2)
plt.hist(y, density=True)
plt.show()

## Problem 3: Konfidensintervall (forts.)
# Plotta histogrammet och skattningen.
plt.figure()
plt.hist(y, density=True)
plt.plot(lower_bound, 0.6, 'g*', markersize=10)
plt.plot(upper_bound, 0.6, 'g*', markersize=10)
# Plotta täthetsfunktionen med den skattade parametern.
x_grid = np.linspace(np.min(y), np.max(y), 60)
pdf = stats.rayleigh.pdf(x_grid, scale=est)
plt.plot(x_grid, pdf, 'r')
plt.show()