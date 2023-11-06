from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

# Daten für Bitcoin in Euro herunterladen
data = yf.download('BTC-EUR', start='2021-01-01', end=datetime.today().strftime('%Y-%m-%d'))

# 'Close' Preis als Zielvariable verwenden
data = data[['Close']]

# Daten normalisieren
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# Trainingsdatensatz erstellen
train_data = scaled_data[0:int(len(scaled_data)*0.8), :]

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
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Testdatensatz erstellen
test_data = scaled_data[int(len(scaled_data)*0.8) - 60:, :]

# Testdaten in x_test und y_test aufteilen
x_test, y_test = [], data[int(len(scaled_data)*0.8):]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
x_test = np.array(x_test)

# Daten für LSTM-Modell umformen
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Vorhersagen machen
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Diagramm erstellen
plt.figure(figsize=(12,6))
plt.plot(y_test.values, color='blue', label='Actual Bitcoin Price')
plt.plot(predictions , color='red', label='Predicted Bitcoin Price')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()