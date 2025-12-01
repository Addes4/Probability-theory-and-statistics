## Problem 3: Konfidensintervall for Rayleighfordelning

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import Data_and_tools as tools

# Ladda data.
y = np.loadtxt('Data_and_tools/wave_data.dat')

# Plotta en bit av signalen samt histogrammet.
plt.figure(figsize=(4, 8))
plt.subplot(2, 1, 1)
plt.plot(y[:100])
plt.subplot(2, 1, 2)
plt.hist(y, density=True)
plt.show()

# Skatta parametern på samma sätt som i Problem 2.
est = np.sqrt(np.mean(y**2) / 2)

# Ta fram ett konfidensintervall för skattningen.
# För Rayleigh-fördelning: asymptotisk varians är sigma^2/(2n)
# Vi använder normal approximation för 95% konfidensintervall
alpha = 0.05
n = len(y)
# Standard error
se = est / np.sqrt(2 * n)
# Kvantil för normal approximation
z_alpha_2 = stats.norm.ppf(1 - alpha / 2)
# Konfidensintervall
lower_bound = est - z_alpha_2 * se
upper_bound = est + z_alpha_2 * se

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