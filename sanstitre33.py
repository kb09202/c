# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 01:09:57 2024

@author: pc
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random

# Génération de données fictives
def generate_customer_data(num_customers):
    """
    Génère des données fictives pour les clients.
    """
    data = []
    for i in range(num_customers):
        frequency = random.randint(1, 50)  # Nombre de visites
        spending = random.randint(100, 2000)  # Dépenses totales
        data.append({'customer_id': i + 1, 'frequency': frequency, 'spending': spending})
    return pd.DataFrame(data)

# Générer les données
customer_data = generate_customer_data(100)

# Appliquer le clustering
X = customer_data[['frequency', 'spending']]
kmeans = KMeans(n_clusters=3, random_state=42)
customer_data['cluster'] = kmeans.fit_predict(X)

# Visualisation des clusters
plt.figure(figsize=(10, 6))
for cluster in customer_data['cluster'].unique():
    cluster_data = customer_data[customer_data['cluster'] == cluster]
    plt.scatter(cluster_data['frequency'], cluster_data['spending'], label=f'Cluster {cluster}')
plt.xlabel('Fréquence des visites')
plt.ylabel('Dépenses totales')
plt.title("Segmentation des clients")
plt.legend()
plt.grid(True)
plt.show()
