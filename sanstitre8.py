# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 09:26:08 2024

@author: pc
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Préparer les données
data['Shifted_Close'] = data['Close'].shift(-1)  # Valeur cible décalée
df = data.dropna()
X = df[['Close']].values
y = df['Shifted_Close'].values

# Diviser les données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ajuster le modèle Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Faire des prévisions
y_pred = model.predict(X_test)

# Visualiser
plt.figure(figsize=(10, 6))
plt.plot(y_test, label="Vraies Valeurs")
plt.plot(y_pred, label="Prédictions", color='red')
plt.legend()
plt.title("Prévision avec Random Forest")
plt.show()
