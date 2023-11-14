from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler


# Funktion zum Herunterladen der Daten
def download_data(symbol, start_date):
    """Download the data for the given symbol from start date to today."""
    return yf.download(symbol, start=start_date, end=datetime.today().strftime('%Y-%m-%d'))


# Funktion zur Normalisierung der Daten
def normalize_data(data):
    """Normalize data using MinMaxScaler."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(data), scaler


# Funktion zum Erstellen des Trainingsdatensatzes
def create_dataset(data, time_step=60):
    """Create dataset for training the LSTM model."""
    x, y = [], []
    for i in range(time_step, len(data)):
        x.append(data[i - time_step:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)


# Funktion zum Umformen der Daten für das LSTM-Modell
def reshape_for_lstm(data):
    """Reshape the data for LSTM model (samples, time steps, features)."""
    return np.reshape(data, (data.shape[0], data.shape[1], 1))


# Funktion zum Erstellen des LSTM-Modells
def build_lstm_model(input_shape):
    """Build LSTM model with specified input shape."""
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    return model


# Funktion zum Erstellen der Vorhersagen
def predict_prices(model, x_test, scaler):
    """Make predictions using the trained LSTM model and inverse transform the normalized values."""
    predictions = model.predict(x_test)
    return scaler.inverse_transform(predictions)


# Funktion zum Zeichnen des Diagramms
def plot_predictions(actual, predicted):
    """Plot the actual vs predicted prices."""
    plt.figure(figsize=(12, 6))
    plt.plot(actual, color='blue', label='Actual Bitcoin Price')
    plt.plot(predicted, color='red', label='Predicted Bitcoin Price')
    plt.title('Bitcoin Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


# Skript-Hauptteil
data = download_data('BTC-EUR', '2021-01-01')
data = data[['Close']]  # 'Close' Preis als Zielvariable verwenden
scaled_data, scaler = normalize_data(data)  # Daten normalisieren
train_data = scaled_data[0:int(len(scaled_data) * 0.8), :]  # Trainingsdatensatz erstellen
x_train, y_train = create_dataset(train_data)  # Trainingsdaten in x_train und y_train aufteilen
x_train = reshape_for_lstm(x_train)  # Daten für LSTM-Modell umformen
model = build_lstm_model((x_train.shape[1], 1))  # LSTM-Modell erstellen
model.compile(optimizer='adam', loss='mean_squared_error')  # Modell kompilieren
model.fit(x_train, y_train, batch_size=1, epochs=1)  # Modell trainieren
test_data = scaled_data[int(len(scaled_data) * 0.8) - 60:, :]  # Testdatensatz erstellen
x_test, _ = create_dataset(test_data)  # Testdaten in x_test und y_test aufteilen
x_test = reshape_for_lstm(x_test)  # Daten für LSTM-Modell umformen
predictions = predict_prices(model, x_test, scaler)  # Vorhersagen machen
plot_predictions(data[int(len(scaled_data) * 0.8):]['Close'].values, predictions)  # Diagramm erstellen
