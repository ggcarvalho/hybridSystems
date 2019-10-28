
######################################## IMPORTING ############################
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import pmdarima as pm
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] =15, 6
import seaborn as sns
sns.set()

##############################################################################

def incrementar_serie(serie_real, serie_diff):
    return serie_real[0:-1] + serie_diff

##########################  PLOTS  ############################################
lnmx = pd.read_csv("lnmx_series.csv",index_col="Year")
lnmx = lnmx["40"]
lnmx = lnmx[lnmx.index<2011]

plt.plot(lnmx,linewidth=1,color="k")
plt.xlabel("Time (Years)")
plt.ylabel("Mortality Rate")
plt.title("LNMX Series")
plt.show()



split= 1990

Train, Test = lnmx[lnmx.index<=split] ,lnmx[lnmx.index>split] # Train and Test series
plt.plot(Train,color="darkblue",linewidth=1,label="Train")
plt.plot(Test,color="coral",linewidth=1,label="Test")
plt.legend()
plt.show()
# print(len(lnmx))
# print(len(Train))
# print(len(Test))
# print(  len(lnmx) - len(Train) - len(Test)    )

lnmx_diff = lnmx.diff().dropna()
plt.plot(lnmx_diff,c="k",linewidth=1)
plt.xlabel("Time (Years)")
plt.title(" LNMX Differentiated ")
plt.show()

# # # ACF plot
# # plot_acf(lnmx, lags=20, c="k")
# # plt.show()

# # # PACF plot
# # plot_pacf(lnmx, lags=20, c= "k")
# # plt.show()

# ACF plot
plot_acf(lnmx_diff, lags=30, c="k")
plt.title("ACF (diff)")
plt.show()

# PACF plot
plot_pacf(lnmx_diff, lags=30, c= "k")
plt.title("PACF (diff)")
plt.show()



# # model = pm.auto_arima(Train, start_p=0, start_q=0,
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
fitted.plot_predict(dynamic=False,start=1,end=len(Train))
plt.title("ARIMA(1,1,1) -- Train")
plt.legend(['ARIMA fit', 'Train' ])
plt.show()


plt.plot(Test,label="Test",color="coral",linewidth=1)
plt.plot(fc_series, label='Forecast',color="blue",linewidth=1)
plt.title("ARIMA(1,1,1) -- Test")
L=plt.legend(loc='best')
L.get_texts()[0].set_text('Test')
L.get_texts()[1].set_text('Forecast')
plt.show()


# MSE
mse_arima_train = mean_squared_error(lnmx_diff[lnmx_diff.index<=split], arima_fittedValues)
print("MSE ARIMA Train (diff. serie)= ", mse_arima_train)
mse_arima = mean_squared_error(Test, fc_series)
print("MSE ARIMA Test= ", mse_arima)
#print(fitted.summary())
#plot residual errors
residuals = pd.DataFrame(fitted.resid)
plt.plot(residuals,color='k',linewidth=1)
plt.title("ARIMA(1,1,1) residuals")
plt.show()
plot_acf(residuals,color="k",lags=30)
plt.title("ACF ARIMA(1,1,1) residuals")
plt.show()
plot_pacf(residuals,color="k",lags=30)
plt.title("PACF ARIMA(1,1,1) residuals")
plt.show()
residuals.plot(kind='kde',color='k').get_legend().remove()
plt.show()
print(residuals.describe())
