## Problem 3: Linjär regression med logaritmisk transformation
## Modell: w_k = log(y_k) = β_0 + β_1 * x_k + ε_k

import numpy as np
import matplotlib.pyplot as plt
from Data_and_tools.tools import regress

# Ladda datafilen
birth = np.loadtxt('Data_and_tools/birth.dat')

# Extrahera variabler
# y: barnets födelsevikt (kolonn 3, index 2)
y = birth[:, 2]

# x: oberoende variabel - exempelvis moderns ålder (kolonn 4, index 3)
# Du kan också använda moderns längd (index 15) eller moderns vikt (index 14)
x = birth[:, 3]  # Moderns ålder

# Filtrera bort NaN-värden
# Vi behöver par där både x och y är giltiga
valid_mask = ~(np.isnan(x) | np.isnan(y))
x = x[valid_mask]
y = y[valid_mask]

print("Linjär regression med logaritmisk transformation")
print("Modell: log(y_k) = β_0 + β_1 * x_k + ε_k")
print(f"\nAntal observationer: {len(x)}")
print(f"Oberoende variabel (x): Moderns ålder")
print(f"Beroende variabel (y): Barnets födelsevikt (gram)")

# Steg 1: Transformera beroende variabeln
# w_k = log(y_k)
w = np.log(y)

print(f"\nTransformation: w = log(y)")
print(f"w min: {np.min(w):.4f}, w max: {np.max(w):.4f}")

# Steg 2: Skapa designmatris X
# För modellen w = β_0 + β_1 * x behöver vi:
# X = [1, x_1; 1, x_2; ...; 1, x_n]
# Kolonn 1: ettor för intercept (β_0)
# Kolonn 2: x-värden för lutning (β_1)
X = np.column_stack([np.ones(len(x)), x])

print(f"\nDesignmatris X har form: {X.shape}")
print("  - Första kolonnen: ettor (för intercept β_0)")
print("  - Andra kolonnen: x-värden (för lutning β_1)")

# Steg 3: Använd tools.regress för att skatta parametrar
# Signifikansnivå: 5% (alpha = 0.05)
alpha = 0.05
beta, beta_int = regress(X, w, alpha=alpha)

# Steg 4: Presentera resultat
print(f"Skattade parametrar:")
print(f"β_0 (intercept): {beta[0]:.6f}")
print(f"β_1 (lutning):   {beta[1]:.6f}")

print(f"Konfidensintervall ({(1-alpha)*100}%):")
print(f"β_0: [{beta_int[0, 0]:.6f}, {beta_int[0, 1]:.6f}]")
print(f"β_1: [{beta_int[1, 0]:.6f}, {beta_int[1, 1]:.6f}]")

# Tolka modellen
print(f"Tolkning:")
print(f"log(födelsevikt) = {beta[0]:.4f} + {beta[1]:.4f} * ålder")
print(f"eller:")
print(f"födelsevikt = exp({beta[0]:.4f} + {beta[1]:.4f} * ålder)")

print("Slutsats:")
print(f"Modellen log(y) = {beta[0]:.4f} + {beta[1]:.4f} * x beskriver")
print(f"sambandet mellan moderns ålder och logaritmen av födelsevikten.")
if beta[1] > 0:
    print(f"Lutningen är positiv ({beta[1]:.4f}), vilket indikerar att")
    print(f"födelsevikten ökar med moderns ålder (i logaritmisk skala).")
else:
    print(f"Lutningen är negativ ({beta[1]:.4f}), vilket indikerar att")
    print(f"födelsevikten minskar med moderns ålder (i logaritmisk skala).")

