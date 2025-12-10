## Problem 4: Fördelningar av givna data

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import Data_and_tools as tools

# Laddar datafilen
birth = np.loadtxt('Data_and_tools/birth.dat')

# Skapar histogram för olika variabler, vi tar ut kolonn 3, 4, 15 och 16, 
# minus 1 för rätt index enligt python

# 1. Barnets födelsevikt (kolonn 3, index 2)
birth_weight = birth[:, 2]
# Filtrera bort NaN-värden
birth_weight = birth_weight[~np.isnan(birth_weight)]

# 2. Moderns ålder (kolonn 4, index 3)
mother_age = birth[:, 3]
# Filtrera bort NaN-värden
mother_age = mother_age[~np.isnan(mother_age)]

# 3. Moderns längd (kolonn 16, index 15)
mother_height = birth[:, 15]
# Filtrera bort NaN-värden
mother_height = mother_height[~np.isnan(mother_height)]

# 4. Moderns vikt (kolonn 15, index 14)
mother_weight = birth[:, 14]
# Filtrera bort NaN-värden
mother_weight = mother_weight[~np.isnan(mother_weight)]

# Skapa figur med fyra histogram
plt.figure(figsize=(12, 10))

# Histogram 1: Barnets födelsevikt
plt.subplot(2, 2, 1)
plt.hist(birth_weight, bins=40, density=True, edgecolor='black')
plt.xlabel('Födelsevikt (gram)')
plt.ylabel('Täthet')
plt.title('Barnets födelsevikt')
plt.grid(True, alpha=0.3)

# Histogram 2: Moderns ålder
plt.subplot(2, 2, 2)
plt.hist(mother_age, bins=26, density=True, edgecolor='black')
plt.xlabel('Ålder (år)')
plt.ylabel('Täthet')
plt.title('Moderns ålder')
plt.grid(True, alpha=0.3)

# Histogram 3: Moderns längd
plt.subplot(2, 2, 3)
plt.hist(mother_height, bins=40, density=True, edgecolor='black')
plt.xlabel('Längd (cm)')
plt.ylabel('Täthet')
plt.title('Moderns längd')
plt.grid(True, alpha=0.3)

# Histogram 4: Moderns vikt
plt.subplot(2, 2, 4)
plt.hist(mother_weight, bins=40, density=True, edgecolor='black')
plt.xlabel('Vikt (kg)')
plt.ylabel('Täthet')
plt.title('Moderns vikt')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Filtrerar och sorterar utifrån ifall mödrarna röker eller ej, värde 3 är 
# rökare och < 3 är icke rökare
non_smokers = (birth[:, 19] < 3)
smokers = (birth[:, 19] == 3)

# Extraherar födelsevikten för de två kategorierna.
# Filtrerar bort NaN-värden
x = birth[non_smokers, 2]
x = x[~np.isnan(x)]
y = birth[smokers, 2]
y = y[~np.isnan(y)]


# Skapar en stor figur som plottar datan på
plt.figure(figsize=(8, 8))
# Plotta ett låddiagram över icke-rökare (x).
plt.subplot(2, 2, 1)
plt.boxplot(x, labels=['Icke-rökare'])
plt.ylabel('Födelsevikt (gram)')
plt.title('Födelsevikt: Icke-rökare')
plt.axis([0, 2, 500, 5000])
plt.grid(True, alpha=0.3)

# Plotta ett låddiagram över rökare (y).
plt.subplot(2, 2, 2)
plt.boxplot(y, labels=['Rökare'])
plt.ylabel('Födelsevikt (gram)')
plt.title('Födelsevikt: Rökare')
plt.axis([0, 2, 500, 5000])
plt.grid(True, alpha=0.3)

# Beräkna kärnestimator för x och y. Funktionen
# gaussian_kde returnerar ett funktionsobjekt som sedan
# kan evalueras i godtyckliga punkter.
kde_x = stats.gaussian_kde(x)
kde_y = stats.gaussian_kde(y)

# Skapar ett rutnät för vikterna som vi kan använda för att
# beräkna kärnestimatorernas värden.
# Använd birth_weight som redan är definierad och filtrerad från NaN
min_val = np.min(birth_weight)
max_val = np.max(birth_weight)
grid = np.linspace(min_val, max_val, 60)

# Plotta kärnestimatorerna.
plt.subplot(2, 2, (3, 4))
plt.plot(grid, kde_x(grid), 'b', label='Icke-rökare', linewidth=2)
plt.plot(grid, kde_y(grid), 'r', label='Rökare', linewidth=2)
plt.xlabel('Födelsevikt (gram)')
plt.ylabel('Täthet')
plt.title('Jämförelse: Kärnestimatorer för födelsevikt')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
label_x = 'Kärnestimator för födelsevikt för icke-rökare'


# Extra undersökning, jämför kvinnor som dricker under graviditeten 
# med de som aldrig dricker eller slutade när de blev gravida

# Tar fram alla de som dricker eller inte dricker
non_drinkers = (birth[:, 25] < 2)
drinkers = (birth[:, 25] == 2)

# Definierar variabler vi kan använda för att ta fram kärnestimatorer
z2 = birth[drinkers, 2]
z2 = z2[~np.isnan(z2)]
z = birth[non_drinkers, 2]
z = z[~np.isnan(z)]

# Tar fram kärnestimatorer
kde_z = stats.gaussian_kde(z)
kde_z2 = stats.gaussian_kde(z2)

# Plottar de grafer vi vill ha, två lådagram och en stor linjegraf
plt.subplot(2, 2, 1)
plt.boxplot(z, labels=['Non-drinkers'])
plt.ylabel('Födelsevikt (gram)')
plt.title('Födelsevikt: Non-drinkers')
plt.axis([0, 2, 500, 5000])
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.boxplot(z2, labels=['Drinkers'])
plt.ylabel('Födelsevikt (gram)')
plt.title('Födelsevikt: Drinkers')
plt.axis([0, 2, 500, 5000])
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, (3, 4))
plt.plot(grid, kde_z(grid), 'b', label='Non-drinkers', linewidth=2)
plt.plot(grid, kde_z2(grid), 'r', label='Drinkers', linewidth=2)
plt.xlabel('Födelsevikt (gram)')
plt.ylabel('Täthet')
plt.title('Jämförelse: Kärnestimatorer för födelsevikt')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
label_x = 'Kärnestimator för födelsevikt för non-drinkers'

#Skriver ut hur många kvinnor från varje grupp, alltså drickande och ickedricanke
antal_nondrinker = np.shape(z2)
antal_drinker = np.shape(z)
print('Hej hej', antal_nondrinker, antal_drinker)
