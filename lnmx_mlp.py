
################################### IMPORTING #######################################################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error as MSE
import statsmodels.api as sm
from matplotlib.pylab import rcParams
import pandas as pd
import seaborn as sns
sns.set()
rcParams['figure.figsize'] =15, 6

#####################################################################################################

lnmx = pd.read_csv("lnmx_series.csv",index_col="Year")
lnmx = lnmx["40"]
dados = lnmx[lnmx.index<2013]
serie = dados

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


def select_lag_acf(serie, max_lag):
    from statsmodels.tsa.stattools import acf
    x = serie[0: max_lag+1]

    acf_x, confint = acf(serie, nlags=max_lag, alpha=.05, fft=False,
                             unbiased=False)

    limiar_superior = confint[:, 1] - acf_x
    limiar_inferior = confint[:, 0] - acf_x

    lags_selecionados = []

    for i in range(1, max_lag+1):


        if acf_x[i] >= limiar_superior[i] or acf_x[i] <= limiar_inferior[i]:
            lags_selecionados.append(i-1)  #-1 por conta que o lag 1 em python é o 0

    #caso nenhum lag seja selecionado, essa atividade de seleção
    # para o gridsearch encontrar a melhor combinação de lags
    if len(lags_selecionados)==0:


        print('NENHUM LAG POR ACF')
        lags_selecionados = [i for i in range(max_lag)]

    print('LAGS', lags_selecionados)

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #inverte o valor dos lags para usar na lista de dados se os dados forem de ordem [t t+1 t+2 t+3]
    lags_selecionados = [max_lag - (i+1) for i in lags_selecionados]
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    return lags_selecionados

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
tam_janela = 2
serie_janelas = gerar_janelas(tam_janela, serie)
x_train, y_train, x_test, y_test, x_val, y_val = split_serie_with_lags(serie_janelas, 0.5897,
 perc_val = 0.3077)



def treinar_mlp(x_train, y_train, x_val, y_val,num_exec):


    neuronios =  [1,2,3,5,10]  #[1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 170, 200]
    func_activation =  ['tanh','relu']#['tanh']   #['identity', 'tanh', 'relu']
    alg_treinamento = ['lbfgs','adam','sgd']#, 'sgd', 'adam']
    max_iteracoes = [10000] #[100, 1000, 10000]
    learning_rate = ['constant','adaptive']  #['constant', 'invscaling', 'adaptive']
    qtd_lags_sel = len(x_train[0])
    best_result = np.Inf
    for i in range(0,len(neuronios)):
        for j in range(0,len(func_activation)):
            for l in range(0,len(alg_treinamento)):
                for m in range(0,len(max_iteracoes)):
                    for n in range(0,len(learning_rate)):
                        for qtd_lag in range(1, len(x_train[0])+1):
                        #variar a qtd de pontos utilizados na janela

                            print('QTD de Lags:', qtd_lag, 'Qtd de Neuronios' ,neuronios[i],
                             'Func. Act', func_activation[j])


                            for e in range(0,num_exec):
                                mlp = MLPRegressor(hidden_layer_sizes=neuronios[i],
                                 activation=func_activation[j], solver=alg_treinamento[l],
                                  max_iter = max_iteracoes[m], learning_rate= learning_rate[n])


                                mlp.fit(x_train[:,-qtd_lag:], y_train)
                                predict_validation = mlp.predict(x_val[:,-qtd_lag:])
                                mse = MSE(y_val, predict_validation)

                                if mse < best_result:
                                    best_result = mse
                                    print('Melhor MSE:', best_result)
                                    select_model = mlp
                                    qtd_lags_sel = qtd_lag


    return select_model, qtd_lags_sel

modelo, lag_sel = treinar_mlp(x_train, y_train, x_val, y_val,10)

predict_train = modelo.predict(x_train[:, -lag_sel:])
predict_val = modelo.predict(x_val[:, -lag_sel:])
predict_test = modelo.predict(x_test[:, -lag_sel:])

previsoes_train = np.hstack(( predict_train, predict_val))
target_train = np.hstack((y_train, y_val))

plt.plot(previsoes_train, label = 'Forecast: Train + Validation')
plt.plot(target_train, label='Train + Validation')
plt.legend(loc='best')
plt.show()

plt.plot(predict_test, label = 'Forecast Test')
plt.plot(y_test, label='Test')
plt.legend(loc='best')
plt.show()

print("MSE treinamento = %s" %MSE(previsoes_train,target_train))
print("MSE Teste = %s" %MSE(y_test, predict_test))
print(modelo)
print(len(predict_test))