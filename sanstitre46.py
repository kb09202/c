# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 04:52:42 2024

@author: pc
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Génération de données réseau fictives (normal et anomalies)
np.random.seed(42)
normal_data = np.random.normal(loc=0.5, scale=0.1, size=(1000, 10))  # Comportement normal
anomalous_data = np.random.uniform(low=0.1, high=0.9, size=(50, 10))  # Anomalies

# Combiner et étiqueter les données
data = np.vstack([normal_data, anomalous_data])
labels = np.array([0] * len(normal_data) + [1] * len(anomalous_data))  # 0: Normal, 1: Anomalie

# Normalisation des données
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Séparation en jeu d'entraînement et de test
train_data = data_scaled[:800]
test_data = data_scaled[800:]
test_labels = labels[800:]

# Construction de l'autoencodeur
autoencoder = Sequential([
    Dense(8, activation='relu', input_shape=(10,)),
    Dense(4, activation='relu'),
    Dense(8, activation='relu'),
    Dense(10, activation='sigmoid')
])

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

# Entraînement du modèle sur les données normales uniquement
autoencoder.fit(train_data, train_data, epochs=50, batch_size=32, shuffle=True, verbose=1)

# Détection des anomalies
reconstructed_data = autoencoder.predict(test_data)
reconstruction_error = np.mean(np.square(test_data - reconstructed_data), axis=1)

# Définir un seuil d'anomalie
threshold = np.percentile(reconstruction_error, 95)
print("Seuil d'anomalie :", threshold)

# Identifier les anomalies
anomalies = reconstruction_error > threshold
print("Anomalies détectées :", np.sum(anomalies))

# Visualisation des erreurs de reconstruction
plt.figure(figsize=(10, 6))
plt.hist(reconstruction_error, bins=50, alpha=0.7, label="Erreurs de reconstruction")
plt.axvline(threshold, color='r', linestyle='--', label="Seuil d'anomalie")
plt.legend()
plt.xlabel("Erreur de reconstruction")
plt.ylabel("Fréquence")
plt.title("Détection des anomalies dans les données réseau")
plt.show()
