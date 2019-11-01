import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error as MSE
from pmdarima import auto_arima
import seaborn as sns
sns.set()
np.random.seed(1)
def gerar_janelas(tam_janela, serie):
    # serie: vetor do tipo numpy ou lista
    tam_serie = len(serie)
    tam_janela = tam_janela +1 # Adicionado mais um ponto para retornar o target na janela

    janela = list(serie[0:0+tam_janela]) #primeira janela p criar o objeto np
    janelas_np = np.array(np.transpose(janela))

    for i in range(1, tam_serie-tam_janela):
        janela = list(serie[i:i+tam_janela])
        j_np = np.array(np.transpose(janela))

        janelas_np = np.vstack((janelas_np, j_np))

    return janelas_np

def split_serie_with_lags(serie, perc_train, perc_val = 0):

    #faz corte na serie com as janelas já formadas

    x_date = serie[:, 0:-1]
    y_date = serie[:, -1]

    train_size = np.fix(len(serie) *perc_train)
    train_size = train_size.astype(int)

    if perc_val > 0:
        val_size = np.fix(len(serie) *perc_val).astype(int)


        x_train = x_date[0:train_size,:]
        y_train = y_date[0:train_size]
        print("Particao de Treinamento:", 0, train_size  )

        x_val = x_date[train_size:train_size+val_size,:]
        y_val = y_date[train_size:train_size+val_size]

        print("Particao de Validacao:",train_size, train_size+val_size)

        x_test = x_date[(train_size+val_size):-1,:]
        y_test = y_date[(train_size+val_size):-1]

        print("Particao de Teste:", train_size+val_size, len(y_date))

        return x_train, y_train, x_test, y_test, x_val, y_val

    else:

        x_train = x_date[0:train_size,:]
        y_train = y_date[0:train_size]

        x_test = x_date[train_size:-1,:]
        y_test = y_date[train_size:-1]

        return x_train, y_train, x_test, y_test

# importing data
lnmx_series = pd.read_csv('lnmx_series.csv', sep = ',', index_col = 0)

idades = lnmx_series.columns

lnmx = lnmx_series['40']

treino = lnmx.loc[:1990]
teste = lnmx.loc[1991:2011]
lnmx = lnmx.loc[:2011]

stepwise_model = auto_arima(treino, start_p=1, start_q=1,
                           max_p=10, max_q=10, m=12,
                           start_P=0, seasonal=False,
                           d=1, D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)

# treinamento do arima(1,1,1)
stepwise_model.fit(treino)

# extração do fit e plotando o gráfico
arima_fit_treino = stepwise_model.predict_in_sample()
arima_fit_treino = pd.Series(arima_fit_treino, index = treino.iloc[:-1].index)
#treino.plot(label='Real')
#arima_fit_treino.plot(label='Fitted arima(1,1,1)')
#plt.legend()
#plt.show()

# extração dos resíduos e plotando o gráfico
resid_arima = stepwise_model.resid()
resid_arima = pd.Series(resid_arima, index = treino.iloc[:-1].index)
#resid_arima.plot(label='resíduos arima')
#plt.legend()
#plt.show()

# Criando as Janelas com 2 lags
lags = 2
janelas = gerar_janelas(tam_janela=lags, serie=resid_arima)

def treinar_mlp(x_train, y_train, x_val, y_val, num_exec):


    neuronios =  [1, 2, 3, 5, 10]    #[1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 170, 200]
    func_activation =  ['tanh', 'relu']   #['identity', 'tanh', 'relu']
    alg_treinamento = ['lbfgs', 'sgd', 'adam']#, ['lbfgs','sgd', 'adam']
    max_iteracoes = [10000] #[100, 1000, 10000]
    learning_rate = ['adaptive']  #['constant', 'invscaling', 'adaptive']
    qtd_lags_sel = len(x_train[0])
    best_result = np.Inf
    for i in range(0,len(neuronios)):
        for j in range(0,len(func_activation)):
            for l in range(0,len(alg_treinamento)):
                for m in range(0,len(max_iteracoes)):
                    for n in range(0,len(learning_rate)):
                        for qtd_lag in range(1, len(x_train[0]+1)): #variar a qtd de pontos utilizados na janela

                            print('QTD de Lags:', qtd_lag, 'Qtd de Neuronios' ,neuronios[i], 'Func. Act', func_activation[j])


                            for e in range(0,num_exec):
                                mlp = MLPRegressor(hidden_layer_sizes=neuronios[i], activation=func_activation[j], solver=alg_treinamento[l], max_iter = max_iteracoes[m], learning_rate= learning_rate[n])


                                mlp.fit(x_train[:,-qtd_lag:], y_train)
                                predict_validation = mlp.predict(x_val[:,-qtd_lag:])
                                rmse = np.sqrt(MSE(y_val, predict_validation))

                                if rmse < best_result:
                                    best_result = rmse
                                    print('Melhor RMSE:', best_result)
                                    select_model = mlp
                                    qtd_lags_sel = qtd_lag


    return select_model, qtd_lags_sel

def split_serie_with_lags(serie, perc_train, perc_val = 0):

    #faz corte na serie com as janelas já formadas

    x_date = serie[:, 0:-1]
    y_date = serie[:, -1]

    train_size = np.fix(len(serie) *perc_train)
    train_size = train_size.astype(int)

    if perc_val > 0:
        val_size = np.fix(len(serie) *perc_val).astype(int)


        x_train = x_date[0:train_size,:]
        y_train = y_date[0:train_size]
        print("Particao de Treinamento:", 0, train_size  )

        x_val = x_date[train_size:train_size+val_size,:]
        y_val = y_date[train_size:train_size+val_size]

        print("Particao de Validacao:",train_size, train_size+val_size)

        x_test = x_date[(train_size+val_size):-1,:]
        y_test = y_date[(train_size+val_size):-1]

        print("Particao de Teste:", train_size+val_size, len(y_date))

        return x_train, y_train, x_test, y_test, x_val, y_val

    else:

        x_train = x_date[0:train_size,:]
        y_train = y_date[0:train_size]

        x_test = x_date[train_size:-1,:]
        y_test = y_date[train_size:-1]

        return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test, x_val, y_val = split_serie_with_lags(janelas, perc_train = 0.5897, perc_val = 0.3077)

# treinar MLP
mlp_model, lag_sel = treinar_mlp(x_train, y_train, x_val, y_val, num_exec = 10)

print('Modelo mlp :', mlp_model)
print('Quantidade de lags: ', lag_sel)

# Avaliando o modelo nos dados de X_teste
predict_test = mlp_model.predict(x_test[:, -lag_sel:])

# gráfico da previsão
plt.plot(y_test, label = 'Resíduos do ARIMA\n de treinamento')
plt.plot(predict_test, label = 'Previsão com MLP\n dos Resíduos do Arima\n de treinamento')
plt.legend(loc='best',fontsize="x-small")
plt.show()

# Previsao com o Arima
previsao_arima = stepwise_model.predict(n_periods=20)
previsao_arima = pd.Series(previsao_arima, index = teste.iloc[:-lag_sel].index)

# Calculando a diferença do Arima para o valor da Série
erro_previsao_arima = teste.iloc[:-lag_sel] - previsao_arima

# gerando as janelas dos erros_previsao_arima e incluindo a observação do ano 1990 que está nos dados resid_arima
erro_previsao_arima = np.concatenate((resid_arima.iloc[-lag_sel:].values, erro_previsao_arima))
dados_teste = gerar_janelas(tam_janela=lag_sel-1, serie=erro_previsao_arima)

# Realizando a previsão da mlp no erro do arima
previsao_mlp = mlp_model.predict(dados_teste[:,-lag_sel:])
previsao_mlp = pd.Series(previsao_mlp, index = teste.iloc[:-lag_sel].index)

# Criando o sistema híbrido
z = previsao_arima + previsao_mlp

# plotando
teste.loc[:2010].plot(label = 'Série')
previsao_arima.plot(label = 'ARIMA')
z.plot(label = 'Z = ARIMA + MLP')
plt.legend()
plt.show()

# printando o RMSE
print('RMSE - ARIMA = %s'%np.sqrt(MSE(teste.iloc[:-lag_sel], previsao_arima)))
print('RMSE - Híbrido (z) = %s'%np.sqrt(MSE(y_true = teste.iloc[:-1], y_pred = z)))