# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
### Date: 03-11-2026
### Name: S.Harika
### Reg no: 212224240155
### Dataset: Tesla stock price 
### S/w read: Colab


### AIM:
To implement ARMA model in python.
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
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

data = pd.read_csv('/content/tesla.csv.zip')

print(data.head())

N = 1000

plt.rcParams['figure.figsize'] = [12, 6]

X = data['open'].dropna()

plt.plot(X)
plt.title('Original Data')
plt.xlabel('Time')
plt.ylabel('Open Price')
plt.show()

plt.subplot(2, 1, 1)

plot_acf(X, lags=50, ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)

plot_pacf(X, lags=50, ax=plt.gca())
plt.title('Original Data PACF')

plt.tight_layout()
plt.show()

arma11_model = ARIMA(X, order=(1, 0, 1)).fit()

phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

print("AR parameter:", phi1_arma11)
print("MA parameter:", theta1_arma11)

ar1 = np.array([1, -phi1_arma11])

ma1 = np.array([1, theta1_arma11])

ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_1)
plt.title('ACF of ARMA(1,1)')
plt.show()

plot_pacf(ARMA_1)
plt.title('PACF of ARMA(1,1)')
plt.show()

arma22_model = ARIMA(X, order=(2, 0, 2)).fit()

phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']

theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

ar2 = np.array([1, -phi1_arma22, -phi2_arma22])

ma2 = np.array([1, theta1_arma22, theta2_arma22])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_2)
plt.title('ACF of ARMA(2,2)')
plt.show()

plot_pacf(ARMA_2)
plt.title('PACF of ARMA(2,2)')
plt.show()

```

### OUTPUT:

ORIGINAL DATA:

<img width="1317" height="637" alt="image" src="https://github.com/user-attachments/assets/039b3f64-15f6-43ff-8659-9352ecd2eec0" />

ORIGINAL ACF DATA:

<img width="1612" height="342" alt="image" src="https://github.com/user-attachments/assets/bf82d281-aca1-4d6c-b289-1fe67ca1d109" />

ORIGINAL PACF DATA:

<img width="1505" height="342" alt="image" src="https://github.com/user-attachments/assets/02040dc3-eb78-4c36-a944-f2c38b0d863d" />

### SIMULATED ARMA(1,1) PROCESS:

<img width="1366" height="621" alt="image" src="https://github.com/user-attachments/assets/4b6abd6c-12cf-4102-86ea-a5b7900fc320" />

Partial Autocorrelation:

<img width="1257" height="612" alt="image" src="https://github.com/user-attachments/assets/c80a71b8-4667-4831-84e8-b5998329008a" />


Autocorrelation:

<img width="1247" height="606" alt="image" src="https://github.com/user-attachments/assets/d75842e4-7e7f-4b6c-a017-ab6752722c93" />


### SIMULATED ARMA(2,2) PROCESS:

<img width="1282" height="597" alt="image" src="https://github.com/user-attachments/assets/6796c72e-6d1a-4493-9156-5c0a6325c004" />


Partial Autocorrelation:

<img width="1217" height="601" alt="image" src="https://github.com/user-attachments/assets/00c0a2dc-51ce-4aa2-95c7-b54bbfbbbaf1" />

Autocorrelation:

<img width="1246" height="602" alt="image" src="https://github.com/user-attachments/assets/2006cf44-d6d9-4c33-bf86-e0a1ab819b7c" />


### RESULT:

Thus, a python program is created to fir ARMA Model successfully.
