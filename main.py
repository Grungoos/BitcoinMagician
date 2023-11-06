from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Daten für Bitcoin in Euro herunterladen
data = yf.download('BTC-EUR', start='2016-01-01', end=datetime.today().strftime('%Y-%m-%d'))

# Aktueller Preis
current_price = data['Close'][-1]

# 'Close' Preis als Zielvariable verwenden
data['Prediction'] = data['Close'].shift(-1)

# Die letzten 'n' Zeilen entfernen
n = 30  # Anzahl der Tage, die wir vorhersagen möchten
data = data[:-n]

# Unabhängige Datensatz erstellen
X = np.array(data.drop(['Prediction'], axis=1))

# Zielvariable erstellen
y = np.array(data['Prediction'])

# Daten in Trainings- und Testdaten aufteilen
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Lineares Regressionsmodell erstellen
lr = LinearRegression()

# Modell trainieren
lr.fit(x_train, y_train)

# Modell testen
lr_confidence = lr.score(x_test, y_test)
print("lr confidence: ", lr_confidence)

# 'n' Zeilen des originalen Datensatzes anzeigen
x_forecast = np.array(data.drop(['Prediction'], axis=1))[-n:]

# Vorhersage für die nächsten 'n' Tage
lr_prediction = lr.predict(x_forecast)
print(lr_prediction)

# Diagramm erstellen
plt.figure(figsize=(12,6))
plt.plot(range(len(lr_prediction)), lr_prediction, color='blue', linestyle='-', label='Predicted Price')
plt.scatter(0, current_price, color='red', label='Current Price')  # Aktuellen Preis anzeigen
plt.title('Bitcoin price prediction for next 30 days')
plt.xlabel('Days')
plt.ylabel('Price (€)')
plt.legend()
plt.show()