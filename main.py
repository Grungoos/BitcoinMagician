import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import yfinance as yf
from datetime import datetime

# Daten für Bitcoin in Euro herunterladen
data = yf.download('BTC-EUR', start='2014-01-01', end=datetime.today().strftime('%Y-%m-%d'))

# 'Close' Preis als Zielvariable verwenden
data = data[['Close']]

# Daten normalisieren
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# Trainingsdatensatz erstellen
train_data = scaled_data

# Trainingsdaten in x_train und y_train aufteilen
x_train, y_train = [], []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

# Daten für LSTM-Modell umformen (samples, time steps, features)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# LSTM-Modell erstellen
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Modell kompilieren und trainieren
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=1, epochs=50)  # Anzahl der Epochen auf 50 erhöht

# Vorhersage für die nächsten 'n' Tage
n = 30
predictions = []
for _ in range(n):
    x_input = train_data[-60:]
    x_input = x_input.reshape((1, -1, 1))
    y_hat = model.predict(x_input)
    predictions.append(y_hat[0][0])
    train_data = np.append(train_data, y_hat)
    train_data = train_data[-60:]

predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Diagramm erstellen
plt.figure(figsize=(12,6))
plt.plot(range(len(predictions)), predictions, color='blue', label='Predicted Bitcoin Price')
plt.title('Bitcoin Price Prediction for next 30 days')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()