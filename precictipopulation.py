# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 09:59:03 2024

@author: pc
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --- Étape 1 : Générer des données historiques synthétiques ---
years = np.array([2000, 2005, 2010, 2015, 2020, 2025]).reshape(-1, 1)  # Années
populations = {
    "Afrique": [800, 900, 1000, 1100, 1250, 1400],  # En millions
    "Asie": [3700, 3900, 4100, 4300, 4500, 4700],
    "Europe": [730, 740, 750, 740, 730, 720],
    "Amérique": [900, 930, 960, 990, 1020, 1050],
    "Océanie": [30, 35, 40, 45, 50, 55]
}

# Conversion en DataFrame
data = pd.DataFrame(populations, index=years.flatten())
print("Données historiques :")
print(data)

# --- Étape 2 : Prédictions pour chaque continent ---
future_years = np.array([2030, 2035, 2040, 2045, 2050]).reshape(-1, 1)

predictions = {}
models = {}

for continent, values in populations.items():
    # Créer et entraîner le modèle de régression linéaire
    model = LinearRegression()
    model.fit(years, values)
    models[continent] = model
    
    # Prédire les valeurs futures
    predicted_population = model.predict(future_years)
    predictions[continent] = predicted_population

# --- Étape 3 : Visualisation des résultats ---
plt.figure(figsize=(12, 6))

for continent, values in populations.items():
    # Tracer les données historiques
    plt.plot(years, values, label=f"{continent} (Historique)")
    
    # Tracer les prédictions
    plt.plot(future_years, predictions[continent], '--', label=f"{continent} (Prédictions)")

# Configuration du graphique
plt.title("Prédiction de la population mondiale par continent")
plt.xlabel("Année")
plt.ylabel("Population (en millions)")
plt.legend()
plt.grid()
plt.show()

# --- Résumé des prédictions ---
for continent, predicted_values in predictions.items():
    print(f"\nPrédictions pour {continent} :")
    for year, pop in zip(future_years.flatten(), predicted_values):
        print(f"  Année {year}: {pop:.2f} millions")
