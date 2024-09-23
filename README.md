### Developed By : SHARAN MJ
### Register No. 212222240097
### Date : 


# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES

### AIM:

#### To implement ARMA model in python for Onion Price Data

### ALGORITHM:

1. Import necessary libraries.

2. Set up matplotlib settings for figure size.

3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000
  data points using the ArmaProcess class. Plot the generated time series and set the title and x-
  axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
  plot_acf and plot_pacf.

5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000
  data points using the ArmaProcess class. Plot the generated time series and set the title and x-
  axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
  plot_acf and plot_pacf.

### PROGRAM:

#### Importing Necessary Libraries and Loading the Dataset

```py
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('/content/drive/MyDrive/time series/AMZN.csv')


```

####  Extracting and Plotting the 'Onion Price' Data

```py
# Extracting and cleaning the 'Price of AMAZON STOCK' data
X = df['Low'].replace([np.inf, -np.inf], np.nan).dropna() # Changed 'Min' to 'Low'

# Plot ACF and PACF of the original data before any ARMA modeling
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plot_acf(X, lags=25, ax=plt.gca())
plt.title('Original Price of amazon stock Data ACF')
plt.subplot(1, 2, 2)
plot_pacf(X, lags=25, ax=plt.gca())
plt.title('Original Price of amzon stock Data PACF')
plt.tight_layout()
plt.show()

```
#### Train-Test Split and ARMA Model Fitting

```py
# Train-Test Split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]

# Fit ARMA(1,1) model to the 'Price' data to estimate parameters
arma11_model = ARIMA(X_train, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

# Simulate ARMA(1,1) process
ar_params_11 = np.array([1, -phi1_arma11])  # AR(1) parameter
ma_params_11 = np.array([1, theta1_arma11])  # MA(1) parameter
arma11_process = ArmaProcess(ar_params_11, ma_params_11)

# Set random seed and simulate ARMA(1,1)
np.random.seed(42)  # For reproducibility
simulated_arma11 = arma11_process.generate_sample(nsample=len(X))

```

#### Plotting Simulated ARMA(1,1) Data

```py
# Plot ACF, PACF, and Simulated ARMA(1,1)
plt.figure(figsize=(18, 6))

# Simulated ARMA(1,1) ACF and PACF
plt.subplot(1, 3, 1)
plot_acf(simulated_arma11, lags=25, ax=plt.gca())
plt.title('Simulated ARMA(1,1) ACF')
plt.subplot(1, 3, 2)
plot_pacf(simulated_arma11, lags=25, ax=plt.gca())
plt.title('Simulated ARMA(1,1) PACF')

# Simulated ARMA(1,1) plot
plt.subplot(1, 3, 3)
plt.plot(simulated_arma11, label='Simulated ARMA(1,1)', color='orange')
plt.title('Simulated ARMA(1,1) Data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()


```

#### ARMA(2,2) Model Fitting and Simulation

```py

# Fit ARMA(2,2) model to the 'AMAZON STOCK' data to estimate parameters
arma22_model = ARIMA(X_train, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

# Simulate ARMA(2,2) process
ar_params_22 = np.array([1, -phi1_arma22, -phi2_arma22])  # AR(2) parameters
ma_params_22 = np.array([1, theta1_arma22, theta2_arma22])  # MA(2) parameters
arma22_process = ArmaProcess(ar_params_22, ma_params_22)

# Set random seed and simulate ARMA(2,2)
np.random.seed(42)  # For reproducibility
simulated_arma22 = arma22_process.generate_sample(nsample=len(X))


```

#### Plotting Simulated ARMA(2,2) Data

```py
# Plot ACF, PACF, and Simulated ARMA(2,2)
plt.figure(figsize=(18, 6))

# Simulated ARMA(2,2) ACF and PACF
plt.subplot(1, 3, 1)
plot_acf(simulated_arma22, lags=25, ax=plt.gca())
plt.title('Simulated ARMA(2,2) ACF')
plt.subplot(1, 3, 2)
plot_pacf(simulated_arma22, lags=25, ax=plt.gca())
plt.title('Simulated ARMA(2,2) PACF')

# Simulated ARMA(2,2) plot
plt.subplot(1, 3, 3)
plt.plot(simulated_arma22, label='Simulated ARMA(2,2)', color='green')
plt.title('Simulated ARMA(2,2) Data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()

```


### OUTPUT:

#### Autocorrelation and Partial Autocorrelation

![Screenshot 2024-09-23 114419](https://github.com/user-attachments/assets/aaee084f-07f3-4c7c-8d0b-8bcbcbccfa95)



#### SIMULATED ARMA(1,1) PROCESS: Autocorrelation and Partial Autocorrelation


![Screenshot 2024-09-23 114409](https://github.com/user-attachments/assets/140603c0-f6f0-4f5b-aac8-bdc825434298)




SIMULATED ARMA(2,2) PROCESS: Autocorrelation and Partial Autocorrelation


![Screenshot 2024-09-23 114357](https://github.com/user-attachments/assets/de3ad240-3ea3-47ea-9c7d-4dac049d06f4)



### RESULT:

#### Thus, a python program is created to for ARMA Model successfully.
