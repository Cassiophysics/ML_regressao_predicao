# Machine Learning - Regressão Linear

![capa_ml_rl](https://github.com/Cassiophysics/ML_regressao_predicao_passagens_aereas/assets/108491443/bc728550-b582-4001-a9e6-65f67e3bc4bc)


## **Teste você mesmo o modelo**: [✈️ PREVISOR DE PREÇOS DE PASSAGENS AÉREAS](https://cassiophysics-ml-regressao-predicao-passage-streamlitapp-dhb08h.streamlit.app/)

Este é um projeto de Machine Learning de aprendizado supervisionado e o problema é do tipo regressão, onde se tem como objetivo treinar um algoritmo para encontrar uma relação linear entre os dados de entrada e saída e assim fazer uma previsão.

Para tal propósito, o conjunto de dados utilizado foi obtido a partir do site [KAGGLE](https://www.kaggle.com/datasets/nikhilmittal/flight-fare-prediction-mh) e contém diversas características a fim de se prever o preço de passagens aéreas.

## Motivação:

Partindo do pressuposto de que é difícil prever os preços das passagens aéreas de forma precisa e confiável, pois sabemos que os preços são altamente variáveis e dependem de diversos fatores, como data, horário, destino, sazonalidade, demanda, entre outros.

Este projeto tem como objetivo atender à demanda por soluções mais eficazes na precificação de passagens aéreas, buscando melhorar a experiência do usuário ao planejar viagens e identificar oportunidades de negócio no setor de turismo e aviação.

A elaboração de um modelo de machine learning para previsão de preços pode trazer benefícios significativos para companhias aéreas, agências de viagem e consumidores. Por exemplo:

**Precificação otimizada:** Com um modelo de previsão de preços preciso, as empresas podem ajustar seus preços de forma otimizada, levando em consideração fatores como sazonalidade, demanda, oferta e concorrência. Isso pode resultar em uma melhor estratégia de precificação, maximizando o lucro e a competitividade.

**Melhor gestão de capacidade:** O modelo de previsão de preços pode ajudar as empresas a gerenciar melhor a capacidade dos voos. Com base nas previsões de demanda, as empresas podem tomar decisões informadas sobre a frequência de voos, capacidade de assentos e planejamento de rotas. Isso pode ajudar a evitar voos subutilizados ou superlotados, resultando em uma melhor alocação de recursos e redução de custos.

**Otimização de promoções e ofertas:** As empresas podem identificar momentos ideais para lançar promoções e ofertas especiais. Ao analisar padrões de demanda e preços históricos, o modelo pode sugerir os momentos em que os clientes são mais sensíveis a descontos e ofertas atrativas. Isso pode ajudar a aumentar a demanda em períodos de baixa procura e maximizar a receita em períodos de alta demanda.

**Previsão de receitas:** Além de prever os preços das passagens, o modelo também pode ser usado para prever a receita total gerada por determinados voos, rotas ou períodos de tempo. Essas previsões podem ser usadas para fins de planejamento financeiro, alocação de recursos e tomada de decisões estratégicas.

**Análise de concorrência:** O modelo pode fornecer insights valiosos sobre a dinâmica competitiva do mercado de passagens aéreas. Ao analisar os preços praticados pela concorrência e compará-los com as próprias estratégias de precificação, as empresas podem ajustar suas táticas de mercado e permanecer competitivas.

**Melhoria da experiência do cliente:** Ao prever os preços das passagens aéreas com maior precisão, a empresa pode fornecer uma experiência mais transparente e confiável para os clientes. Isso ajuda a criar confiança e satisfação, pois os clientes sabem que estão obtendo preços justos e competitivos.

Diante disso, este projeto tem como objetivos específicos elaborar um modelo capaz de auxiliar na tomada de decisão nessas questões e apresentá-lo por meio de uma interface gráfica utilizando o Streamlit.

## Estrutura do Projeto:

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

Foi realizada uma análise da multicolinearidade por meio do Variance Inflation Factor (VIF). A multicolinearidade ocorre quando as variáveis independentes são muito correlacionadas umas com as outras. O fator de inflação de variância VIF identifica a correlação entre as variáveis independentes e a força dessa correlação. Se VIF > 5 é um motivo de preocupação e se VIF > 10 indica um sério problema de colinearidade.
Embora a multicolinearidade não tenha grande influência na capacidade preditiva de um modelo, afeta diretamente nas pontuações de importância dos preditores usados para construir o modelo, quanto mais variáveis correlacionadas tivermos, menor a importância de todas elas simultaneamente, é como se as variáveis colineares repartissem a importância entre elas. Assim, o objetivo de se usar o VIF neste projeto é apenas verificar a multicolinearidade para uma seleção de variáveis mais precisa.

## **4. Seleção de Variáveis**

A técnica utilizada foi o Extra Tree Regressor, esta é uma abordagem baseada em modelo para selecionar os recursos usando os modelos supervisionados baseados em árvore para tomar decisões sobre a importância dos recursos. O Extra Tree Regressor é um algoritmo de conjunto que semeia vários modelos de árvores construídos aleatoriamente a partir do conjunto de dados de treinamento e classifica os recursos que foram mais votados. Ele ajusta cada árvore de decisão em todo o conjunto de dados em vez de uma réplica bootstrap e escolhe um ponto de divisão aleatoriamente para dividir os nós. A divisão de nós, que ocorre em todos os níveis das árvores de decisão constituintes, é baseada na medida de aleatoriedade ou entropia nos sub-nós. Os nós são divididos em todas as variáveis disponíveis no conjunto de dados e a divisão que resulta no subconjunto mais homogêneo é selecionada nos modelos de árvore constituinte. Isso reduz a variância e torna o modelo menos propenso a overfitting.

## **5. Modelos Baseline com Cross Validation**
Inicialmente avaliamos vários modelos de regressão sem qualquer hiperparâmetro através da Cross Validation. A métrica escolhida para avaliação foi o R-quadrado(R2), uma medida estatística que representa a proporção da variância de uma variável dependente que é explicada por uma variável ou variáveis independentes em problemas de regressão. Assim, quanto maior o R², mais explicativo é o modelo linear, ou seja, melhor ele se ajusta à amostra.
Também avaliamos os modelos LinearRegression, KNeighborsRegressor e SVR, algoritmos que não são baseados em árvores, com dados normalizados e padronizados para verificar se superam os resultados dos demais algoritmos.


## **6. Pré-processamento para os Dados de Teste**
Nesta etapa foi feito para os dados de teste os mesmos procedimentos que fizemos com os dados de treino

## **7. Comparação dos Melhores Modelos Avaliados por Cross Validation**
Avaliação dos 3 melhores modelos via Cross Validation com os dados de teste. As métricas utilizadas para a avaliação dos modelos foram:

R2 - Expressa a quantidade da variância dos dados que é explicada pelo modelo construído.

MAE - Consiste na média da diferença absoluta entre valores preditos e reais.

MSE - Consiste na média do erro das previsões ao quadrado. Pega-se a diferença entre o valor predito pelo modelo e o valor real, eleva-se o resultado ao quadrado, faz-se a mesma coisa com todos os outros pontos, soma-os, e divide-se pelo número de elementos preditos. Quanto maior esse número, pior o modelo.

RMSE - Tendo em vista a diferença de unidades, o RMSE entra como uma forma de melhorar a interpretabilidade da métrica MSE tirando a raiz quadrada e acertando a unidade. Entretanto, essa medida, assim como o MSE, penaliza predições muito distantes do real.

MAPE - Exprime uma porcentagem, obtida através da divisão da diferença entre predito e real pelo valor real. Assim como o MSE e o MAE, quanto menor o valor, mais preciso seria o modelo de regressão.

## **8. Otimização dos Modelos - Ajuste de Hiperparâmetros com BayesSearchCV**
Foi realizada a otimização de hiperparâmetros, no qual refere-se à realização de uma pesquisa para descobrir o conjunto de argumentos de configuração de modelo específico que resultam no melhor desempenho do modelo em um conjunto de dados específico. A técnica empregada foi BayesSearchCV, a otimização Bayesiana leva em consideração as avaliações anteriores ao escolher o conjunto de hiperparâmetros a ser avaliado a seguir. Ao escolher suas combinações de parâmetros de maneira informada, ele se permite focar nas áreas do espaço de parâmetros que acredita trazer as pontuações de validação mais promissoras. Essa abordagem geralmente requer menos iterações para obter o conjunto ideal de valores de hiperparâmetros. Principalmente porque desconsidera as áreas do espaço de parâmetros que acredita não trazer nada de relevante.
Isso, por sua vez, limita o número de vezes que um modelo precisa ser treinado para validação, pois apenas as configurações que devem gerar uma pontuação de validação mais alta são passadas para avaliação.

## Impacto nos negócios:

Podemos mensurar o impacto que a implementação desse modelo pode trazer para uma empresa comparando a receita, o lucro, os custos operacionais, a eficiência, a satisfação do cliente, a vantagem competitiva, entre outras métricas financeiras relevantes para o negócio, antes e depois do uso do modelo. Além disso, o Retorno sobre Investimento (ROI) que consiste na diferença entre os ganhos obtidos com o modelo e os custos do projeto, dividido pelo custo do projeto.

## Identificação de melhorias para o modelo:

**Refinar o modelo:** podemos continuar aperfeiçoando o modelo, adicionando novas variáveis relevantes para o modelo e o negócio. Isso pode ser feito em parceria com as equipes de marketing e vendas, através do compartilhamento de insights e informações valiosas. Além disso, podemos utilizar técnicas mais avançadas, como deep learning, e realizar testes e validações frequentemente para garantir a qualidade do modelo.

**Integração com sistemas internos:** ao fazer a conexão entre o modelo e os sistemas internos de uma empresa, como sistemas de reservas e gerenciamento, podemos implementar as estratégias de precificação de forma mais eficiente e atualizar os preços em tempo real, de acordo com as informações fornecidas pelo modelo.

**Monitoramento contínuo:** é importante acompanhar as métricas do modelo de forma frequente para identificar oportunidades de melhoria e fazer os ajustes necessários de forma otimizada. Isso permite que o modelo permaneça atualizado e eficaz ao longo do tempo.


