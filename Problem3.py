## Problem 3: Konfidensintervall for Rayleighfordelning

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import Data_and_tools as tools

# Laddar data
y = np.loadtxt('Data_and_tools/wave_data.dat')

# Plotta en bit av signalen samt histogrammet
plt.figure(figsize=(4, 8))
plt.subplot(2, 1, 1)
plt.plot(y[:100])
plt.subplot(2, 1, 2)
plt.hist(y, 40, density=True)
plt.show()

# Skatta parametern på samma sätt som i Problem 2.
# ML-skattning för att plotta PDF:en
est = np.sqrt(np.mean(y**2) / 2)

# Ta fram ett konfidensintervall för skattningen.
est_mk = np.sqrt(2 / np.pi) * np.mean(y)

# Vi använder normal approximation för 95% konfidensintervall
ALPHA = 0.05
n = len(y)

# Kvantil för normal approximation (z_alpha/2)
z_alpha_2 = stats.norm.ppf(1 - ALPHA / 2)
# Vi multiplicerar in estimatet för b direkt här, 
# den är utanför rottecknet så vi höjer inte upp den med 2, 
# därför kan man tro att den inte är lik uttrycket vi räknade ut i förberedelse uppgiften
marginal_error = z_alpha_2 * est_mk * np.sqrt((4 - np.pi) / (np.pi * n))
# Konfidensintervall
lower_bound = est_mk - marginal_error
upper_bound = est_mk + marginal_error


# Plottar histogrammet och skattningen.
plt.figure()
plt.hist(y, 40,  density=True)
plt.plot(lower_bound, 0.6, 'g*', markersize=10)
plt.plot(upper_bound, 0.6, 'g*', markersize=10)
# Plottar täthetsfunktionen med den skattade parametern.
x_grid = np.linspace(np.min(y), np.max(y), 60)
pdf = stats.rayleigh.pdf(x_grid, scale=est)
plt.plot(x_grid, pdf, 'r')
plt.show()