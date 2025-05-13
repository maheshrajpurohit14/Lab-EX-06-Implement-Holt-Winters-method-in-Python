# Lab-EX-06-Implement-Holt-Winters-method-in-Python

## AIM:
To implement the Holt Winters Method Model using Python.

## ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as datetime, set it as index, and perform some initial data exploration
3. Resample it to a monthly frequency beginning of the month
4. You plot the time series data, and determine whether it has additive/multiplicative
trend/seasonality
5. Split test,train data,create a model using Holt-Winters method, train with train data and Evaluate the model predictions against test data
6. Create teh final model and predict future data and plot it

## PROGRAM: 

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from math import sqrt

data = pd.read_csv('football.csv')

data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')
data.dropna(subset=['Date'], inplace=True)
data.set_index('Date', inplace=True)

<ipython-input-4-1ac6354a9395>:1: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')

monthly_sales = data['Team Score'].resample('MS').sum()

plt.figure(figsize=(12, 6))
monthly_sales.plot(title="Score")
plt.xlabel('Date')
plt.ylabel('Score')
plt.grid(True)
plt.show()


decomposition = seasonal_decompose(monthly_sales, model='additive', period=12)
decomposition.plot()
plt.suptitle("Additive Decomposition", fontsize=16)
plt.tight_layout()
plt.show()

train = monthly_sales[:-12]
test = monthly_sales[-12:]

model = ExponentialSmoothing(
train,
trend='add',
seasonal='add',
seasonal_periods=12,
initialization_method='estimated'
)
model_fit = model.fit()

predictions = model_fit.forecast(12)

rmse = sqrt(mean_squared_error(test, predictions))
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

final_model = ExponentialSmoothing(
monthly_sales,
trend='add',
seasonal='add',
seasonal_periods=12,
initialization_method='estimated'
).fit()

forecast = final_model.forecast(12)

plt.figure(figsize=(14, 7))
monthly_sales.plot(label='Actual Score')
forecast.plot(label='Forecast', color='red', linestyle='--')
plt.title('Holt-Winters Forecast - Score')
plt.xlabel('Date')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.show()

print(f'Mean Sales: {monthly_sales.mean():.2f}')
print(f'Standard Deviation of Sales: {monthly_sales.std():.2f}')

```


## OUTPUT:

### TEST PREDICTION:

![image](https://github.com/user-attachments/assets/6a6bd770-4ae7-41f6-9715-c9987269039d)

### FINAL PREDICTGION:

![image](https://github.com/user-attachments/assets/4ceb8900-90a1-490b-accf-b80fa88a6428)


## RESULT:
Thus the program run successfully based on the Holt Winters Method model.
