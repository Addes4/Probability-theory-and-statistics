
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import Data_and_tools as tools

# Laddar datafilen
birth = np.loadtxt('Data_and_tools/birth.dat')

# Extrahera variablerna (samma som i 4:an)
# 1. Barnets födelsevikt (kolonn 3, index 2)
birth_weight = birth[:, 2]
birth_weight = birth_weight[~np.isnan(birth_weight)]

# 2. Moderns ålder (kolonn 4, index 3)
mother_age = birth[:, 3]
mother_age = mother_age[~np.isnan(mother_age)]

# 3. Moderns längd (kolonn 16, index 15)
mother_height = birth[:, 15]
mother_height = mother_height[~np.isnan(mother_height)]

# 4. Moderns vikt (kolonn 15, index 14)
mother_weight = birth[:, 14]
mother_weight = mother_weight[~np.isnan(mother_weight)]

# För att använda Prob plot sorterar vi bort alla NaN 
# för annars syns inte den röda linjen

# Skapa figur med fyra probplots
plt.figure(figsize=(12, 10))

# Probplot 1: Barnets födelsevikt
plt.subplot(2, 2, 1)
_ = stats.probplot(birth_weight, plot=plt)
plt.title('Barnets födelsevikt')
plt.grid(True, alpha=0.3)

# Probplot 2: Moderns ålder
plt.subplot(2, 2, 2)
_ = stats.probplot(mother_age, plot=plt)
plt.title('Moderns ålder')
plt.grid(True, alpha=0.3)

# Probplot 3: Moderns längd
plt.subplot(2, 2, 3)
_ = stats.probplot(mother_height, plot=plt)
plt.title('Moderns längd')
plt.grid(True, alpha=0.3)

# Probplot 4: Moderns vikt
plt.subplot(2, 2, 4)
_ = stats.probplot(mother_weight, plot=plt)
plt.title('Moderns vikt')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Testar ifall alla är normalfördelade genom Jarque Bera test
# Signifikansnivå: 5% (alpha = 0.05)
ALPHA = 0.05

# Testvariabler
variables = {
    'Barnets födelsevikt': birth_weight,
    'Moderns ålder': mother_age,
    'Moderns längd': mother_height,
    'Moderns vikt': mother_weight
}

# Kritisk värde för chi-kvadrat med 2 frihetsgrader
chi2_critical = stats.chi2.ppf(1 - ALPHA, df=2)

print(f"\nKritiskt värde (X^2): {chi2_critical:.4f}")


# Utför test för varje variabel
for var_name, data in variables.items():
    # Jarque-Bera test
    jb_stat, p_value = stats.jarque_bera(data)
    
    # Beslut: Om p-värde < alpha, förkasta H0 (data är inte normalfördelat)
    is_normal = p_value >= ALPHA
    
    print(f"\n{var_name}:")
    print(f"  p-värde: {p_value:.6f}")
    
    if is_normal:
        print(f"  p är större än alfa (p ≥ {ALPHA})")
    else:
        print(f"  p är mindre än alfa (p < {ALPHA})")
    
    # Beräkna skevhet och kurtosis för att beskriva avvikelsen
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)  # Excess kurtosis (normal = 0)
    
    print(f"  Skewness (gamma): {skewness:.4f} (normal = 0)")
    print(f"  Kurtosis (kappa): {kurtosis:.4f} (normal = 0)")