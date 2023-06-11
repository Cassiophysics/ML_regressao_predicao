# Importação das Bibliotecas

# Manipulação
import pandas as pd
import numpy as np
from scipy import stats

# Machine Learning

import sklearn
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

import joblib


# Carregamento dos dados
df = pd.read_csv('flight_price.csv')
df = df.drop(columns=df.columns[0])

# Separação das variáveis preditoras da variável alvo
X = df.drop(columns='preco') 
Y = df['preco']

# Divisão do Dataset em treino e test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=18)

# Transformação das variáveis categóricas em numéricas
X_train = pd.get_dummies(X_train, prefix=['linha', 'origem', 'destino'], columns=['linha', 'origem', 'destino'])

# Resetar os índices do Dataset
X_train.reset_index(drop=True, inplace=True)

# Criação do DataFrame com as colunas finais selecionadas
X_train = pd.DataFrame(X_train, 
             columns=['horas_total_duracao', 'linha_Jet Airways', 'dia_voo', 'mes_voo', 'linha_Jet Airways Business',
'linha_Multiple carriers', 'linha_Air India', 'destino_Delhi', 'min_partida', 'hora_chegada',
'hora_partida', 'min_chegada', 'destino_New Delhi', 'destino_Hyderabad', 'origem_Mumbai',
'linha_IndiGo', 'origem_Banglore', 'linha_Vistara', 'origem_Delhi'])

# Pré-processamento para os Dados de Teste

#DUMMIES

X_test = pd.get_dummies(X_test, prefix=['linha', 'origem', 'destino'], columns=['linha', 'origem','destino'])

# Criação do DataFrame para X_test
X_test = pd.DataFrame(X_test, 
             columns=['horas_total_duracao', 'linha_Jet Airways', 'dia_voo', 'mes_voo', 'linha_Jet Airways Business',
'linha_Multiple carriers', 'linha_Air India', 'destino_Delhi', 'min_partida', 'hora_chegada',
'hora_partida', 'min_chegada', 'destino_New Delhi', 'destino_Hyderabad', 'origem_Mumbai',
'linha_IndiGo', 'origem_Banglore', 'linha_Vistara', 'origem_Delhi'])

# Reset dos índices
X_test.reset_index(drop=True, inplace=True)

# Modelo Final

xgbr2 = XGBRegressor(random_state=18, colsample_bytree=0.68,
                                             learning_rate=0.05, max_depth=6,
                                             min_child_weight=1, n_estimators=500,
                                             nthread=-1, subsample=0.74)

xgbr2.fit(X_train, Y_train)

Y_pred2 = xgbr2.predict(X_test)

# Métricas

RSquared = metrics.r2_score(Y_test, Y_pred2)*100
MAE = metrics.mean_absolute_error(Y_test, Y_pred2)
MSE = metrics.mean_squared_error(Y_test, Y_pred2)
RMSE = np.sqrt(metrics.mean_squared_error(Y_test, Y_pred2))
MAPE = metrics.mean_absolute_percentage_error(Y_test, Y_pred2)*100
    
# Resultado Final
    
model_result2 = {'Modelo': ['XGBRegressor'],
                 'R-Squared': [RSquared],
                 'MAE': [MAE],
                 'MSE': [MSE],
                 'RMSE': [RMSE],
                 'MAPE': [MAPE],}

result_final = pd.DataFrame(model_result2, index=['0'])

result_final

# Salvar o Modelo Final
import pickle
with open('modelo_xgbr2.pkl', 'wb') as file:
    pickle.dump(xgbr2, file)






