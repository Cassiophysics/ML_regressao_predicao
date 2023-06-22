# Machine Learning - Regressão Linear

![capa_ml_rl](https://github.com/Cassiophysics/ML_regressao_predicao_passagens_aereas/assets/108491443/bc728550-b582-4001-a9e6-65f67e3bc4bc)


## **Teste você mesmo o modelo**: [✈️ PREVISOR DE PREÇOS DE PASSAGENS AÉREAS](https://cassiophysics-ml-regressao-predicao-passage-streamlitapp-dhb08h.streamlit.app/)

Este é um projeto de Machine Learning de aprendizado supervisionado e o problema é do tipo regressão, onde se tem como objetivo treinar um algoritmo para encontrar uma relação linear entre os dados de entrada e saída e assim fazer uma previsão.

Para tal propósito, o conjunto de dados utilizado foi obtido a partir do site [KAGGLE](https://www.kaggle.com/datasets/nikhilmittal/flight-fare-prediction-mh) e contém diversas características a fim de se prever o preço de passagens aéreas.

O raciocínio utilizado para este projeto foi sistematizado em:

1. Análise Exploratória

2. Pré-processamento

3. Multicolinearidade - Variance Inflation Factor (VIF)

4. Seleção de Variáveis

5. Modelos Baseline com Cross Validation

6. Pré-processamento para os Dados de Teste

7. Comparação dos Melhores Modelos Avaliados por Cross Validation

8. Otimização dos Modelos - Ajuste de Hiperparâmetros com BayesSearchCV

## **1. Análise Exploratória**

Através de diversos elementos visuais diferentes buscou-se examinar e investigar os dados previamente para se ter uma visão panorâmica e obter uma compreensão mais elucidativa dos dados antes da elaboração do modelo.

ALGUMAS CONCLUSÕES:

- A maior parte dos preços está em uma faixa de até 20000, mas existe a presença de outliers.

- A Linha aérea Jet Airways é a mais frequente. Contudo, Jet Airways Business apresenta preço médio muito superior às demais linhas.

- Delhi é de onde parte a maior quantidade dos voos e possui maior preço médio.

- Cochin é o destino com maior número de voos. No entanto, New Delhi é o destino com maior preço médio.

- Um pouco mais da metade dos voos possuem apenas uma parada e quanto maior o número de paradas maior é o preço médio.

- Na coluna informações, a grande maioria não contém informações e as demais não são relevantes para o modelo.

## **2. Pré-processamento**
- Exclusão de campos duplicados
- Exclusão de valores nulos
- Engenharia de Atributos
- Divisão em Dados de Treino e Teste
- Encoding

## **3. Multicolinearidade - Variance Inflation Factor (VIF)**

- Foi realizada uma análise da multicolinearidade por meio do Variance Inflation Factor (VIF). A multicolinearidade ocorre quando as variáveis independentes são muito correlacionadas umas com as outras. O fator de inflação de variância VIF identifica a correlação entre as variáveis independentes e a força dessa correlação. Se VIF > 5 é um motivo de preocupação e se VIF > 10 indica um sério problema de colinearidade.
Embora a multicolinearidade não tenha grande influência na capacidade preditiva de um modelo, afeta diretamente nas pontuações de importância dos preditores usados para construir o modelo, quanto mais variáveis correlacionadas tivermos, menor a importância de todas elas simultaneamente, é como se as variáveis colineares repartissem a importância entre elas. Assim, o objetivo de se usar o VIF neste projeto é apenas verificar a multicolinearidade para uma seleção de variáveis mais precisa.

## **4. Seleção de Variáveis**

- A técnica utilizada foi o Extra Tree Regressor, esta é uma abordagem baseada em modelo para selecionar os recursos usando os modelos supervisionados baseados em árvore para tomar decisões sobre a importância dos recursos. O Extra Tree Regressor é um algoritmo de conjunto que semeia vários modelos de árvores construídos aleatoriamente a partir do conjunto de dados de treinamento e classifica os recursos que foram mais votados. Ele ajusta cada árvore de decisão em todo o conjunto de dados em vez de uma réplica bootstrap e escolhe um ponto de divisão aleatoriamente para dividir os nós. A divisão de nós, que ocorre em todos os níveis das árvores de decisão constituintes, é baseada na medida de aleatoriedade ou entropia nos sub-nós. Os nós são divididos em todas as variáveis disponíveis no conjunto de dados e a divisão que resulta no subconjunto mais homogêneo é selecionada nos modelos de árvore constituinte. Isso reduz a variância e torna o modelo menos propenso a overfitting.

## **5. Modelos Baseline com Cross Validation**
- Inicialmente avaliamos vários modelos de regressão sem qualquer hiperparâmetro através da Cross Validation. A métrica escolhida para avaliação foi o R-quadrado(R2), uma medida estatística que representa a proporção da variância de uma variável dependente que é explicada por uma variável ou variáveis independentes em problemas de regressão. Assim, quanto maior o R², mais explicativo é o modelo linear, ou seja, melhor ele se ajusta à amostra.
Também avaliamos os modelos LinearRegression, KNeighborsRegressor e SVR, algoritmos que não são baseados em árvores, com dados normalizados e padronizados para verificar se superam os resultados dos demais algoritmos.


## **6. Pré-processamento para os Dados de Teste**
- Nesta etapa foi feito para os dados de teste os mesmos procedimentos que fizemos com os dados de treino

## **7. Comparação dos Melhores Modelos Avaliados por Cross Validation**
- Avaliação dos 3 melhores modelos via Cross Validation com os dados de teste. As métricas utilizadas para a avaliação dos modelos foram:

R2 - Expressa a quantidade da variância dos dados que é explicada pelo modelo construído.

MAE - Consiste na média da diferença absoluta entre valores preditos e reais.

MSE - Consiste na média do erro das previsões ao quadrado. Pega-se a diferença entre o valor predito pelo modelo e o valor real, eleva-se o resultado ao quadrado, faz-se a mesma coisa com todos os outros pontos, soma-os, e divide-se pelo número de elementos preditos. Quanto maior esse número, pior o modelo.

RMSE - Tendo em vista a diferença de unidades, o RMSE entra como uma forma de melhorar a interpretabilidade da métrica MSE tirando a raiz quadrada e acertando a unidade. Entretanto, essa medida, assim como o MSE, penaliza predições muito distantes do real.

MAPE - Exprime uma porcentagem, obtida através da divisão da diferença entre predito e real pelo valor real. Assim como o MSE e o MAE, quanto menor o valor, mais preciso seria o modelo de regressão.

## **8. Otimização dos Modelos - Ajuste de Hiperparâmetros com BayesSearchCV**
- Foi realizada a otimização de hiperparâmetros, no qual refere-se à realização de uma pesquisa para descobrir o conjunto de argumentos de configuração de modelo específico que resultam no melhor desempenho do modelo em um conjunto de dados específico. A técnica empregada foi BayesSearchCV, a otimização Bayesiana leva em consideração as avaliações anteriores ao escolher o conjunto de hiperparâmetros a ser avaliado a seguir. Ao escolher suas combinações de parâmetros de maneira informada, ele se permite focar nas áreas do espaço de parâmetros que acredita trazer as pontuações de validação mais promissoras. Essa abordagem geralmente requer menos iterações para obter o conjunto ideal de valores de hiperparâmetros. Principalmente porque desconsidera as áreas do espaço de parâmetros que acredita não trazer nada de relevante.
Isso, por sua vez, limita o número de vezes que um modelo precisa ser treinado para validação, pois apenas as configurações que devem gerar uma pontuação de validação mais alta são passadas para avaliação.
