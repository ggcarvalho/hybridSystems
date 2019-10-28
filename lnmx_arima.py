"""

Bitcoin Time Series Analysis
Author : Gabriel G. Carvalho
Time Series Analysis and Forecast @CInUFPE

"""

######################################## IMPORTING ############################
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import pmdarima as pm
from pykalman import KalmanFilter
from statsmodels.tsa.ar_model import AR
import seaborn as sns
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
from statsmodels.tsa.arima_process import ArmaProcess
from arch import arch_model
######################### FUNCTIONS ###########################################
def returns(serie):
	returns = []
	serie=serie.values
	for i in range(len(serie) -1):
		returns.append(serie[i+1] - serie[i])
	return pd.DataFrame(returns,columns=None)
##########################  PLOTS  ############################################
sns.set()



lnmx = pd.read_csv("lnmx_series.csv") # Using the closing price of
								  # the monthly time series.
lnmx = lnmx["40"]
# plt.plot(lnmx,linewidth=1,color="k")
# plt.xlabel("Time (Years)")
# plt.ylabel("Mortality Rate")
# plt.title("LNMX Series")
# plt.show()



Train, Test = lnmx[:175] ,lnmx[175:195] # Train and Test series
# print(len(lnmx))
# print(len(Train))
# print(len(Test))
# print(  len(lnmx) - len(Train) - len(Test)    )
lnmx_returns = np.diff(lnmx)


plt.plot(lnmx_returns,c="k",linewidth=1)
plt.xlabel("Time (Years)")
plt.ylabel(" LNMX differentiate ")
plt.show()

# # ACF plot
# plot_acf(lnmx, lags=20, c="k")
# plt.show()

# # PACF plot
# plot_pacf(lnmx, lags=20, c= "k")
# plt.show()

# ACF plot
plot_acf(lnmx_returns, lags=30, c="k")
plt.title("ACF (Returns)")
plt.show()

# PACF plot
plot_pacf(lnmx_returns, lags=30, c= "k")
plt.title("PACF (Returns)")
plt.show()

# # model = pm.auto_arima(lnmx, start_p=0, start_q=0,
# #                       test='adf',       # use adftest to find optimal 'd'
# #                       max_p=10, max_q=10, # maximum p and q
# #                       m=1,              # frequency of series
# #                       d=None,           # let model determine 'd'
# #                       seasonal=False,   # No Seasonality
# #                       start_P=0,
# #                       D=0,
# #                       trace=True,
# #                       error_action='ignore',
# #                       suppress_warnings=True,
# #                       stepwise=True)

# # print(model.summary())





# ################################## ARIMA ######################################
# model = ARIMA(train, order=(p,d,q))
model = ARIMA(Train, order=(1, 1, 1))
fitted = model.fit(disp=0)
# Forecast
fc, se, conf = fitted.forecast(len(Test), alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=Test.index)
arima_fittedValues = fitted.fittedvalues
# Plot
fitted.plot_predict(dynamic=False,start=1,end=len(Train)+len(Test))
plt.plot(Test,label="Test",color="k")
plt.plot(fc_series, label='Forecast',color="green")
#plt.title('Forecast vs Actuals')
#plt.legend(loc='upper left', fontsize=8)
plt.title("ARIMA")
L=plt.legend(loc='best')
L.get_texts()[0].set_text('ARIMA Fit')
L.get_texts()[1].set_text('Train')
plt.show()
# MSE
mse_arima_train = mean_squared_error(lnmx_returns[1:175], arima_fittedValues)
print("MSE ARIMA Train (on returns)= ", mse_arima_train)
mse_arima = mean_squared_error(Test, fc_series)
print("MSE ARIMA Test= ", mse_arima)


print(fitted.summary())
#plot residual errors
residuals = pd.DataFrame(fitted.resid)
plt.plot(residuals)
plt.show()
plot_acf(residuals,color="k",lags=30)
plt.title("ACF ARIMA(1,1,1) residuals")
plt.show()
plot_pacf(residuals,color="k",lags=30)
plt.title("PACF ARIMA(1,1,1) residuals")
plt.show()
residuals.plot(kind='kde')
plt.show()
#print(residuals.describe())




