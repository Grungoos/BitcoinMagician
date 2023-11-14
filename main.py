import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yfinance as yf

# Setzen der Umgebungsvariablen für oneDNN (optional, abhängig von Ihrem Bedarf)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def download_data(ticker, start_date):
    # Daten für Bitcoin in Euro herunterladen
    data = yf.download(ticker, start=start_date, end=datetime.today().strftime('%Y-%m-%d'))
    # 'Close' Preis als Zielvariable verwenden
    return data[['Close']]


def normalize_data(data):
    # Daten normalisieren
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler, scaler.fit_transform(data)


def create_train_val_datasets(data, test_size=0.2):
    # Trainings- und Validierungsdaten aufteilen
    return train_test_split(data, test_size=test_size, shuffle=False)


def prepare_lstm_datasets(data, time_steps=60):
    # Trainingsdaten in x und y aufteilen
    x_data, y_data = [], []
    for i in range(time_steps, len(data)):
        x_data.append(data[i - time_steps:i, 0])
        y_data.append(data[i, 0])
    x_data, y_data = np.array(x_data), np.array(y_data)
    return np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1)), y_data


def build_lstm_model(input_shape):
    # LSTM-Modell erstellen mit Dropout und Regularisierung
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape, kernel_regularizer=l1_l2(0.01, 0.01)),
        Dropout(0.2),
        LSTM(50, return_sequences=False, kernel_regularizer=l1_l2(0.01, 0.01)),
        Dropout(0.2),
        Dense(25, kernel_regularizer=l1_l2(0.01, 0.01)),
        Dense(1)
    ])
    # Modell kompilieren
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def fit_model(model, x_train, y_train, x_val, y_val):
    # Early Stopping einrichten
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # Modell trainieren
    return model.fit(x_train, y_train, epochs=100, batch_size=60, validation_data=(x_val, y_val),
                     callbacks=[early_stopping])


def predict_future_values(model, data, scaler, n=30):
    # Vorhersage für die nächsten 'n' Tage
    predictions = []
    current_batch = data[-60:].reshape((1, 60, 1))
    for i in range(n):
        y_hat = model.predict(current_batch, verbose=0)[0][0]
        predictions.append(y_hat)
        current_batch = np.append(current_batch[:, 1:, :], [[[y_hat]]], axis=1)
    # Vorhersagen in den ursprünglichen Wertebereich zurücktransformieren
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))


def plot_predictions(predictions):
    # Diagramm erstellen
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(predictions)), predictions, color='blue', label='Predicted Bitcoin Price')
    plt.title('Bitcoin Price Prediction for next 30 days')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)  # RMSE ist einfach die Wurzel aus MSE
    return mse, mae, rmse


# Anwendung der Funktionen
data = download_data('BTC-EUR', '2014-01-01')
scaler, scaled_data = normalize_data(data)
train_data, val_data = create_train_val_datasets(scaled_data)
x_train, y_train = prepare_lstm_datasets(train_data)
x_val, y_val = prepare_lstm_datasets(val_data)
model = build_lstm_model((x_train.shape[1], 1))
history = fit_model(model, x_train, y_train, x_val, y_val)

# Vorhersagen für Trainings- und Validierungsdaten
y_train_pred = model.predict(x_train)
y_val_pred = model.predict(x_val)

# Denken Sie daran, die Vorhersagen rückzutransformieren, wenn Ihre Zielvariablen normalisiert wurden
y_train_pred = scaler.inverse_transform(y_train_pred)
y_val_pred = scaler.inverse_transform(y_val_pred)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
y_val_actual = scaler.inverse_transform(y_val.reshape(-1, 1))

# Berechnen Sie die Metriken für das Training und die Validierung
train_mse, train_mae, train_rmse = evaluate_model(y_train_actual, y_train_pred)
val_mse, val_mae, val_rmse = evaluate_model(y_val_actual, y_val_pred)

# Ausgabe der Metriken
print(f"Training Data - MSE: {train_mse}, MAE: {train_mae}, RMSE: {train_rmse}")
print(f"Validation Data - MSE: {val_mse}, MAE: {val_mae}, RMSE: {val_rmse}")

# Vorhersagen für die Zukunft
predictions = predict_future_values(model, train_data[-60:], scaler)
plot_predictions(predictions)
