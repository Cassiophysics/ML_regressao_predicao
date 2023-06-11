import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict



#def predict(data):
#    xgbr2 = pickle.load(open('modelo_xgbr2.pkl', 'rb'))
#    return xgbr2.predict(data)

# Carregar o modelo
#with open('modelo_xgbr2.pkl', 'rb') as file:
#    modelo_carregado = pickle.load(file)

#def load_model():
#    with open('/saved_steps.pkl', 'rb') as file:
#        data = pickle.load(file)
#    return data

#data = load_model()

#model_xgbr = data["model"]

# Carregar o Dataset
df = pd.read_csv('/home/cassio/Documentos/MachineLearning_LR/Streamlit/X_train.csv')

# Criar a interface do Streamlit
st.title('✈️ Previsão do Preço de Passagens Aéreas')
st.header('Insira os Dados')

opcoes_linha = st.selectbox('Linha Aérea', ['Jet Airways', 'Jet Airways Business', 'Multiple carriers', 'Air India', 'IndiGo', 'Vistara'])
opcoes_origem = st.selectbox('Origem', ['Bangalore', 'Delhi', 'Mumbai'])
opcoes_destino = st.selectbox('Destino', ['Bangalore', 'Delhi', 'Mumbai'])

dia = st.slider('Dia', 1, 31)
mes = st.slider('Mês', 1, 12)
hora_partida = st.slider('Hora de Partida', 0, 23)
min_partida = st.slider('Minuto de Partida', 0, 59)
hora_chegada = st.slider('Hora de Chegada', 0, 23)
min_chegada = st.slider('Minuto de Chegada', 0, 59)
horas_total_duracao = st.number_input('Horas Totais de Duração', min_value=0.0)

# Realizar a transformação dos dados de entrada
X = pd.DataFrame({
    'horas_total_duracao': [horas_total_duracao],
    'linha_Jet Airways': [opcoes_linha == 'Jet Airways'],
    'dia_voo': [dia],
    'mes_voo': [mes],
    'linha_Jet Airways Business': [opcoes_linha == 'Jet Airways Business'],
    'linha_Multiple carriers': [opcoes_linha == 'Multiple carriers'],
    'linha_Air India': [opcoes_linha == 'Air India'],
    'destino_Delhi': [opcoes_destino == 'Delhi'],
    'min_partida': [min_partida],
    'hora_chegada': [hora_chegada],
    'hora_partida': [hora_partida],
    'min_chegada': [min_chegada],
    'destino_New Delhi': [opcoes_destino == 'New Delhi'],
    'destino_Hyderabad': [opcoes_destino == 'Hyderabad'],
    'origem_Mumbai': [opcoes_origem == 'Mumbai'],
    'linha_IndiGo': [opcoes_linha == 'IndiGo'],
    'origem_Banglore': [opcoes_origem == 'Banglore'],
    'linha_Vistara': [opcoes_linha == 'Vistara'],
    'origem_Delhi': [opcoes_origem == 'Delhi']
})

# Função para fazer a previsão
#def fazer_previsao():
    # Fazer a previsão
#    prediction = model_xgbr.predict(X)
#    return prediction[0]

# Botão de previsão
if st.button('Fazer Previsão'):
    # Calcular e exibir a previsão
    resultado = predict(X)
    st.header('Resultado da Previsão')
    previsao_formatada = np.round(resultado.item(), 2)
    st.write(f'A previsão é: ₹ {previsao_formatada}')


#streamlit run app.py

