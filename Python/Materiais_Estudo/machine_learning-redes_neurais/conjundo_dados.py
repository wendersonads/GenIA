# Objetivos de Aprendizagem
# - Entender a diferença entre dados estruturados e não estruturados;
# - Entender a diferença entre conjunto de dados supervisionados, não supervisionados e semi-supervisionados;
# - Como importar alguns datasets famosos
# - Como fazer a limpeza dos dados
# - Como fazer o pré-processamento
# - Entender a importância da seleção das caratecrísticas
# - Como fazer a divisão dos dados em treino, validação e teste
# - Como fazer a validação cruzada
# - Como tratar dados desbalanceados
# - Entender o sobreajuste ou *overfitting*

# Dados Estruturados e Não Estruturados
# Dados Estruturados
# Os dados estruturados são organizados em um formato específico, como tabelas, **facilitando a sua análise e processamento**.
# Dados Não Estruturados
# Os dados não estruturados não possuem um formato específico e **são mais difíceis de organizar e analisar**.
# Pacotes Python para Dados Estruturados e Não Estruturados
# Dados Estruturados
# Como exemplo de pacotes python para uso com dados estruturados podem ser citados:
# 1. **Pandas**:
# Pandas é um pacote essencial para manipulação e análise de dados estruturados. Ele fornece estruturas de dados como DataFrames e Series, que permitem operações eficientes de filtragem, agregação e transformação de dados tabulares.
# 2. **NumPy**:
# NumPy é uma biblioteca fundamental para computação numérica em Python, oferecendo suporte a arrays multidimensionais e uma ampla gama de funções matemáticas para operações rápidas e eficientes em grandes conjuntos de dados.
# 3. **Scikit-learn**:
# Scikit-learn é uma biblioteca robusta para aprendizado de máquina, que inclui diversas ferramentas para pré-processamento de dados, como normalização, imputação de valores ausentes e codificação de variáveis categóricas, facilitando a preparação de dados estruturados para modelos preditivos.
# 4. **Dask**:
# Dask é uma biblioteca que permite a manipulação de grandes conjuntos de dados que não cabem na memória, dividindo-os em blocos e distribuindo as operações em paralelo. Ele estende a funcionalidade do Pandas para trabalhar com dados maiores do que a memória disponível.
# 5. **SQLAlchemy**:
# SQLAlchemy é uma ferramenta poderosa para a integração de bancos de dados relacionais em Python, facilitando a execução de consultas SQL e a manipulação de dados estruturados diretamente do banco de dados, mantendo a compatibilidade com diferentes sistemas de gerenciamento de banco de dados (SGBDs).
# 6. **Polars**:
# Polars é uma biblioteca de processamento de dados altamente eficiente e de alta performance, projetada para manipulação e análise de grandes conjuntos de dados tabulares. Ele utiliza uma abordagem baseada em colunas e oferece APIs semelhantes ao Pandas, mas com desempenho significativamente mais rápido devido ao seu uso de Rust como backend.
# 7. **PySpark**:
# PySpark é a interface Python para o Apache Spark, um framework de computação distribuída. Ele permite a manipulação e análise de grandes volumes de dados de forma paralela e distribuída, sendo ideal para processamento de Big Data em clusters, com suporte para SQL, machine learning e streaming em tempo real.

# Pandas

# Importação da biblioteca
import pandas as pd

# Definição dos dados de forma estruturada Colunas e Linhas
d = {'col1': [1, 2], 'col2': [3, 4]}

# Importação do dicionário de dados para dentro do DataFrame do pandas
df = pd.DataFrame(data=d)
df

# Dados Não Estruturados
# Como exemplo de pacotes python para uso com dados não estruturados podem ser citados:
# 1. **Scikit-learn**:
# Scikit-learn é uma biblioteca versátil, já abordada também para uso com dados estruturados.  Ela oferece uma ampla gama de algoritmos para aprendizado não supervisionado, como clustering, redução de dimensionalidade e análise de agrupamentos. Ela fornece implementações eficientes de algoritmos como k-means, PCA, e DBSCAN, permitindo a descoberta de padrões em dados não rotulados.
# 2. **OpenCV**:
# OpenCV é uma biblioteca poderosa para processamento de imagens e visão computacional, frequentemente utilizada em tarefas não supervisionadas como segmentação de imagem e detecção de bordas. Sua vasta coleção de ferramentas permite a análise e transformação de imagens, além de suportar operações avançadas de processamento de vídeo.
# 3. **NLTK (Natural Language Toolkit)**:
# NLTK é uma biblioteca essencial para o processamento de linguagem natural (NLP), oferecendo ferramentas para análise de texto não supervisionada, como tokenização, stemming, e análise de frequências. Ela também suporta tarefas como clustering de documentos e análise semântica, facilitando o trabalho com grandes corpora de texto.
# 4. **HDBSCAN**:
# HDBSCAN é uma biblioteca especializada em clustering hierárquico baseado em densidade, que pode identificar clusters de diferentes densidades em dados não supervisionados. Ele é particularmente útil para encontrar agrupamentos em conjuntos de dados complexos e de forma não paramétrica, sem necessidade de definir o número de clusters previamente.
# 5. **Gensim**:
# Gensim é uma biblioteca focada em modelagem de tópicos e análise de grandes corpora de texto. Ela é conhecida por suas implementações de modelos como LDA (Latent Dirichlet Allocation) e Word2Vec, que permitem a extração de tópicos e representações semânticas de documentos em cenários não supervisionados.

# NLTK (Natural Language Toolkit)

# Importação das bibliotecas
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

# Exemplo de texto
texto = "Olá, como você está hoje?"

# Tokenizar o texto
tokens = word_tokenize(texto, language='portuguese')

print(tokens)

# OpenCV

# Importação das bibliotecas
from skimage import io
import matplotlib.pyplot as plt
import cv2

# Carrega a imagem da internet
image = io.imread('https://akcit.ufg.br/_next/image?url=%2Flogo_akcit.png&w=1080&q=75')

# Salva a imagem
io.imsave('image.jpg', image)

# Carrega a imagem salva utilizando o OpenCV
image = cv2.imread('image.jpg')

# Corrige as cores
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.show()

# ----------------------------------------------------------------------

# Conjuntos de Dados Supervisionados, Não Supervisionados e Semi-Supervisionados
# Dados Supervisionados
# Os dados supervisionados são aqueles em que cada exemplo de treinamento é composto por uma entrada e a saída desejada (rótulo). Exemplos incluem:
# - **MNIST**: Conjunto de dados de dígitos manuscritos com rótulos indicando os dígitos de 0 a 9.
# - **Iris**: Conjunto de dados de flores com características como comprimento da pétala e rótulos indicando a espécie da flor.
# - **CIFAR-10**: Conjunto de imagens divididas em 10 classes, como aviões, carros, gatos, etc.
# Exemplo de um conjunto de dados supervisionado
# O conjunto de dados **Iris** é amplamente reconhecido e utilizado em estatística e aprendizado de máquina, especialmente como exemplo introdutório para a aplicação de técnicas de classificação. Este conjunto de dados consiste em 150 amostras de flores pertencentes à espécie **Iris**, distribuídas igualmente entre três subespécies: **Iris setosa**, **Iris versicolor** e **Iris virginica**.
# Cada amostra no conjunto de dados é caracterizada por quatro variáveis numéricas que descrevem atributos morfológicos das flores:
# - **Comprimento da sépala** (em centímetros);
# - **Largura da sépala** (em centímetros);
# - **Comprimento da pétala** (em centímetros);
# - **Largura da pétala** (em centímetros).
# Devido à sua simplicidade, equilíbrio e bem-definidas classes, o conjunto de dados Iris é frequentemente empregado para ilustrar algoritmos de classificação e técnicas de análise exploratória de dados.

# Importa dados do scikitlearn
from sklearn.datasets import load_iris
data = load_iris(as_frame=True)

# Imprime a descrição do dataset
print(data.DESCR)

# Lista de características
data.feature_names

# Lista de rótulos
data.target_names

# Apresenta os dados como Pandas DataFrame
data.frame

# Dados Não Supervisionados
# Os dados não supervisionados são aqueles que não possuem rótulos, e o objetivo é encontrar padrões ou estrutura nos dados. Exemplos incluem:
# - **Google News Word Vectors**: O conjunto de dados consiste em vetores de palavras (word embeddings) treinados em um grande corpus de texto extraído do Google News. Esses vetores foram gerados utilizando o modelo Word2Vec, uma técnica de aprendizado não supervisionado que captura relações semânticas entre palavras ao mapeá-las em um espaço vetorial contínuo. Este conjunto é amplamente utilizado em tarefas de análise semântica, como busca por similaridade de palavras, análise de relações entre palavras, e construção de representações vetoriais para uso em modelos de aprendizado de máquina. Contém cerca de 3 milhões de vetores de palavras e frases, cada um com 300 dimensões.Sua aplicação inclui: Análise semântica, processamento de linguagem natural (NLP), busca por similaridade de palavras. Este conjunto de dados pode ser baixado publicamente em https://code.google.com/archive/p/word2vec/
# - **Análise de Sentimentos de Tweets**: consiste em uma coleção de tweets não rotulados, que podem ser utilizados para tarefas como clustering, redução de dimensionalidade ou identificação de padrões sem supervisão. Embora o conjunto de dados original seja não rotulado, ele é frequentemente utilizado em pesquisas de aprendizado não supervisionado para agrupar tweets com base em sua similaridade ou para extrair temas latentes. Dependendo da coleção possui tamanho variável, mas normalmente composto por milhares ou milhões de tweets.
# É normalmente aplicado em: Clustering, análise de tópicos, redução de dimensionalidade, detecção de tendências em redes sociais.O conjuntos de dados de tweets podem ser extraídos via API ou encontrados em repositórios públicos como o Kaggle em várias competições ou projetos, como por exemplo https://www.kaggle.com/code/leandrodoze/sentiment-analysis-in-portuguese.
# - **Digits Dataset**: é composto por 1,797 imagens de dígitos manuscritos, cada uma com uma resolução de 8x8 pixels, em escala de cinza. Cada imagem representa um dígito de 0 a 9, e é acompanhada por um rótulo que indica o dígito correspondente. Esse conjunto de dados é amplamente utilizado para desenvolver e avaliar algoritmos de aprendizado de máquina em tarefas de reconhecimento de caracteres e classificação de imagens.
# Exemplo de um conjunto de dados não supervisionados
# Como exemplo, será carregado o conjunto de dados **Digits** de dígitos manuscritos utilizando a função `load_digits()` do Scikit-learn e, em seguida, visualiza uma amostra de imagens desse conjunto.
# Primeiro, o conjunto de dados é carregado e os rótulos são removidos para simular um cenário não supervisionado. Em seguida, utiliza-se a biblioteca `Matplotlib` para criar uma grade de subplots e exibir as primeiras 10 imagens do conjunto de dados, cada uma representando um dígito manuscrito em escala de cinza.

# Importar as bibliotecas necessárias
from sklearn import datasets
import matplotlib.pyplot as plt

# Carregar o conjunto de dados de Digits
digits = datasets.load_digits()

# Remover os rótulos para simular um cenário não supervisionado
data = digits.data

# Visualizar algumas imagens do conjunto de dados
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(data[i].reshape(8, 8), cmap='gray')
    ax.axis('off')

plt.show()

# Dados Semi-Supervisionados
# Os dados semi-supervisionados combinam uma pequena quantidade de dados rotulados com uma grande quantidade de dados não rotulados. Na prática esses datasets podem ser adaptados dos dataset supervisionados, retirando os rótulos de alguns caos e assim podendo ser empregado para teste em algoritmos que visam performance em conjunto de dados semi-supervisionados.Exemplos incluem:
# - **Cifar-10 com Rótulos Parciais**: Apenas uma fração das imagens possui rótulos, enquanto o restante é utilizado para melhorar a generalização do modelo.
# - **Dados de Textos Anotados**: Pequeno subconjunto de documentos anotados manualmente, com o restante não anotado, usado em tarefas de classificação de texto.
# - **Vídeos com Quadros Anotados**: Alguns quadros de um vídeo são rotulados, enquanto a maioria não possui rótulos, utilizado para treinamento de modelos de reconhecimento de ações.
# Exemplo de um conjunto de dados semi-supervisionado
# Para criar um dataset semi-supervisionado artificialmente será utilizado o mesmo dataset **Digits** utilizado para o caso não supervisionado.
# Inicialmente, carregam-se os dados e os rótulos associados, depois embaralha aleatoriamente os índices dos exemplos para garantir uma divisão não sistemática. Em seguida, apenas 10% dos rótulos originais são mantidos, sendo que o restante dos exemplos como são rotulados com o valor -1. A partir dessas etapas, pode-se observar como transformar um conjunto de dados rotulado em um formato semi-supervisionado.
# Em `X` estarão os valores dos 64 pixels (imagem de 8x8 pixels) que representam a imagem de um numeral. Em `y` qual é o número de 0 a 9 que aquela imagem representa.

# Importação das bibliotecas
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt

# Carregar o conjunto de dados de dígitos
digits = datasets.load_digits()
# X possui as características da imagem (valores dos 64 pixels --> imagem 8x8)
X = digits.data
# y possui o valor do rótulo (0 a 9) que representa aquela imagem (conjunto de pixels)
y = digits.target

print("Dados das imagens")
print(X)
print("Dados dos rótulos")
print(y)

print("Tamanho dos Dados das imagens:", X.shape)
print("Tamanho dorótulos:", y.shape)

# Quantidade de exemplos por rótulo
y_ori = pd.DataFrame(data=y, columns=['y_ori'])
y_ori.groupby('y_ori')['y_ori'].count()

# Embaralhar o dataset
random_state = np.random.RandomState(42)
indices = np.arange(len(y))
random_state.shuffle(indices)

print("Amostra dos valores de indices: ", indices)
print("Comprimento do indice: ", len(indices))

# Manter rótulos somente de 10% dos dados iniciais
n_labeled_points = int(len(y) * 0.1)
print("Quantidade de rótulos que serão mantidos: ", n_labeled_points)

# Marcar a maioria dos rótulos como -1 (não rotulados)
unlabeled_indices = indices[n_labeled_points:]
y[unlabeled_indices] = -1

# Quantidade de exemplos por rótulo
y_semi = pd.DataFrame(data=y, columns=['y_semi'])
y_semi.groupby('y_semi')['y_semi'].count()

# ----------------------------------------------------------------------

# Limpeza dos dados

# Objetivos
# - Compreender a importância da limpeza de dados.
# - Aprender técnicas básicas de limpeza de dados.
# - Implementar essas técnicas utilizando bibliotecas populares do Python.
# - Praticar a aplicação das técnicas de limpeza de dados em conjuntos de dados reais.

# 1. Introdução
# A limpeza de dados é uma etapa crucial no processo de análise de dados. Dados sujos podem levar a resultados enganosos e comprometer a qualidade da análise. Este notebook irá guiá-lo através dos conceitos e técnicas fundamentais para a limpeza de dados utilizando Python.

# 2. Explicação Teórica
# Importância da Limpeza de Dados
# - **Confiabilidade:** Dados limpos aumentam a confiabilidade das análises.
# - **Qualidade:** Dados de alta qualidade são essenciais para obter insights precisos.
# - **Eficiência:** Dados bem estruturados facilitam a análise e economizam tempo.
# Conceitos Principais
# - **Valores ausentes:** Dados faltantes ou nulos.
# - **Valores duplicados:** Registros idênticos que podem distorcer a análise.
# - **Outliers:** Valores extremos que podem influenciar desproporcionalmente os resultados.
# - **Formatação:** Uniformidade nos formatos de dados.

# 3. Exemplos Práticos
# Nesse exemplo ilustra-se um processo abrangente de limpeza de dados, aplicando técnicas práticas para lidar com valores ausentes, duplicatas e outliers em um conjunto de dados. Primeiramente, o carrega-se um conjunto de dados do Titanic e visualiza-se a presença de valores ausentes utilizando a biblioteca `missingno`, substituindo os valores ausentes na coluna `age` pela média da coluna. Em seguida, verifica-se e remove registros duplicados para garantir a integridade dos dados. Para a detecção e tratamento de outliers na coluna `fare`, o utiliza-se um boxplot para visualizar os outliers e aplica o método do intervalo interquartil (IQR) para remover valores extremos. Finalmente, compara-se a média da coluna `fare` antes e depois da remoção dos outliers, demonstrando o impacto desses valores extremos nos resultados analíticos.

# Carregando Bibliotecas e Dados

import pandas as pd
import numpy as np

# Carregar um conjunto de dados de exemplo
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv'
df = pd.read_csv(url)
df.head()

# Tratamento de Valores Ausentes

# Visualizar valores ausentes
df.isnull().sum()

# Outra maneira de verificar os valores ausentes é visualizando graficamente. Abaixo o código apresenta graficamente os valores ausentes em cada coluna do dataset.
# Em cor escura, os dados preenchidos, na cor branca os valores ausentes. Em alguns casos reais, pode ser que em determinados períodos não houve o registro de certa característica. Porém para chegar a essa conclusão pode não ser tão direto assim. Nesse caso, a visualização gráfica apontaria um grande bloco branco em meio a grandes blocos escuros, e facilitaria essa conclusão.

import missingno as msno

# Visualizar valores ausentes
msno.matrix(df)
plt.show()

# Preencher valores ausentes
df['age'].fillna(df['age'].mean(), inplace=True)

# Visualizar valores ausentes
df.isnull().sum()

# Remoção de Valores Duplicados

# Dimensões do dataset antes da remoção de duplicatas
df.shape

# Verificar as duplicidades
df.loc[df.duplicated()]

# Remover duplicatas
df.drop_duplicates(inplace=True)

# Dimensões do dataset antes da remoção de duplicatas
df.shape

# Tratamento de Outliers
# A técnica apresentada chama-se Tukey's Fence, ela e outras técnicas podem ser encontradas em https://en.wikipedia.org/wiki/Outlier
# O **Tukey's Fence** é um método estatístico utilizado para identificar e remover outliers em um conjunto de dados, particularmente em conjunto com a análise de boxplots. Este método baseia-se em extensões do intervalo interquartil (IQR) e utiliza um critério de "cerca" para determinar quais observações são consideradas outliers.
# O processo envolve calcular o IQR, que é a diferença entre o terceiro quartil (Q3) e o primeiro quartil (Q1), e então aplicar um fator multiplicativo, geralmente 1.5 ou 3, para estabelecer limites superior e inferior para a identificação de outliers. Dados que estão além desses limites são considerados como outliers.
# Em um boxplot, os outliers são visualmente representados como pontos fora das "cercas" do boxplot, facilitando a identificação e a decisão sobre quais valores devem ser removidos para melhorar a qualidade dos dados e a precisão das análises subsequentes.

# Visualizar outliers
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=df['fare'])
plt.show()

# Remover outliers
Q1 = df['fare'].quantile(0.25)
Q3 = df['fare'].quantile(0.75)
IQR = Q3 - Q1
df_sem_outlier = df[~((df['fare'] < (Q1 - 1.5 * IQR)) | (df['fare'] > (Q3 + 1.5 * IQR)))]

# Comparação do efeito dos outliers

print("Média dos valores de fare originais: ", df['fare'].mean())
print("Média dos valores de fare sem outliers: ", df_sem_outlier['fare'].mean())

# 4. Exercícios e Desafios

# Exercício 1: Identificação e Tratamento de Valores Ausentes
# Rode o código abaixo que introduz alguns valores ausentes ao  conjunto de dados de habitação da Califórnia.
# Com base no `df` gerado, utilize o que aprendeu e identifique as colunas com valores ausente com preenchendo com os valores da mediana.

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

# Carregar o conjunto de dados California Housing
data = fetch_california_housing(as_frame=True)
df = data.frame

# Introduzir valores ausentes artificialmente para fins de demonstração
df.loc[::10, 'MedInc'] = np.nan  # Introduzir valores ausentes na coluna 'MedInc'
df.loc[::15, 'HouseAge'] = np.nan  # Introduzir valores ausentes na coluna 'HouseAge'

df.head()

# O código realiza a identificação de colunas numéricas que contêm valores ausentes, preenche esses valores com a mediana de cada coluna e, finalmente, verifica se ainda há valores ausentes no DataFrame.

# Exercício Resolvido
# Identificar colunas numéricas com valores ausentes
# Seleciona todas as colunas do DataFrame df que possuem dados numéricos.
numeric_columns = df.select_dtypes(include=[np.number]).columns

# Identifica quais dessas colunas numéricas possuem valores ausentes.
# O método isnull() é usado para verificar valores ausentes,
# any() verifica se há pelo menos um valor ausente em cada coluna.
# O resultado é uma série de colunas que contêm valores ausentes,
# armazenadas na variável missing_numeric_columns.
missing_numeric_columns = numeric_columns[df[numeric_columns].isnull().any()]

# Apresenta as colunas com valores ausentes
print("Essas são as colunas com valores ausentes: ", missing_numeric_columns)

# Preencher valores ausentes com a mediana
# Percorre cada coluna identificada como contendo valores ausentes.
for column in missing_numeric_columns:
    # Para cada coluna, o método fillna() é utilizado para preencher os valores ausentes da coluna.
    # O método median() calcula a mediana dos valores não ausentes da coluna.
    # O parâmetro inplace=True garante que as mudanças sejam aplicadas diretamente ao DataFrame df,
    # sem a necessidade de criar uma cópia.
    df[column].fillna(df[column].median(), inplace=True)

# Imprime na tela o número de valores ausentes restantes em cada coluna.
# A função isnull() cria uma matriz booleana indicando a presença de valores ausentes,
# sum() calcula o total de valores ausentes por coluna.
print(df.isnull().sum())

# Pré Processamento

# Objetivos
# - **Compreender a importância do pré-processamento de dados.**
# - **Aprender técnicas básicas de normalização e codificação de variáveis categóricas.**
# - **Implementar essas técnicas utilizando bibliotecas populares do Python.**
# - **Praticar a aplicação das técnicas de pré-processamento em conjuntos de dados reais.**

# 1. Introdução
# O pré-processamento de dados é uma etapa essencial no *pipeline* (sequência de etapas) de *machine learning*. Ele prepara os dados para a modelagem, melhorando a eficiência e a precisão dos modelos. Esta aula abordará duas técnicas fundamentais de pré-processamento: normalização e codificação de variáveis categóricas.

# Normalização
# A normalização é o processo de ajustar a escala dos dados para que estejam em um intervalo específico, geralmente entre 0 e 1. Isso é importante porque muitos algoritmos de *machine learning* se beneficiam de dados normalizados, resultando em melhor desempenho e convergência mais rápida.
# Métodos de Normalização
# Saiba mais: https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#plot-all-scaling-minmax-scaler-section
# Aqui serão abordados os 2 principais métodos:
# - **Min-Max Scaling:** Transforma os dados para que seus valores fiquem entre 0 e 1.
# - **Z-score Standardization:** Transforma os dados para que tenham média 0 e desvio padrão 1.
# Exemplo de aplicação de Normalização
# O exemplo abaixo tem como objetivo principal demonstrar técnicas de pré-processamento, especificamente no contexto de um conjunto de dados de imóveis da Califórnia. As etapas envolvem:
# - **Carregamento dos dados**: O conjunto de dados da Califórnia Housing é carregado e armazenado em um DataFrame Pandas para facilitar a manipulação.
# - **Introdução de variáveis categóricas**: Uma nova coluna categórica é criada aleatoriamente para simular uma situação real onde os dados podem conter variáveis desse tipo.
# - **Normalização dos dados**: As variáveis numéricas contínuas 'MedInc' e 'AveOccup' são normalizadas utilizando os métodos MinMaxScaler e StandardScaler, com o objetivo de escalar os valores para um intervalo específico ou com média zero e desvio padrão unitário, respectivamente.
# - **Visualização dos dados**: Gráficos de dispersão são utilizados para visualizar a distribuição dos dados antes e após a normalização.

# Carregando Bibliotecas e Dados

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder

# Carregar o conjunto de dados California Housing
data = fetch_california_housing(as_frame=True)
df = data.frame

# Introduzir variáveis categóricas artificiais

# A função np.random.choice do NumPy é utilizada para gerar amostras aleatórias
# a partir de uma determinada população.
# Ela é usada para criar a coluna 'ocean_proximity', atribuindo aleatoriamente
# uma das três categorias possíveis ('<1H OCEAN', 'INLAND', 'NEAR OCEAN') a cada
# observação do DataFrame.
# em size: O número de amostras a serem geradas (igual ao número de linhas do DataFrame).
# Essa etapa será necessária quando do tratamento de variáveis categóricas.
df['ocean_proximity'] = np.random.choice(['<1H OCEAN', 'INLAND', 'NEAR OCEAN'], size=len(df))

df.head()

# Min-Max Scaling
# - **MedInc**: Representa a renda média dos blocos censitários. Essa variável numérica contínua indica o nível de renda da região onde o imóvel está localizado.
# - **AveOccup**: Representa a ocupação média por unidade habitacional. Essa variável numérica contínua fornece informações sobre a densidade populacional nos imóveis.
# Utilizaremos essas colunas para testar as técnicas de normalização,
# Para melhor compreender, será visto a distribuição desses dados.

# Distribução dos dados antes da normalização
# Atente-se aos valores do eixo x, variando aproximadamente entre 0 a 14
# Atente-se aos valores do eixo y, variando aproximadamente entre 0 e 1200
df.plot.scatter('MedInc', 'AveOccup')

# O `min-max scaling` escala os valores de cada característica para o intervalo [0, 1]. É útil quando se deseja que todas as características tenham a mesma escala.

# Cria um objeto da classe MinMaxScaler.
# Esse objeto armazenará os parâmetros de transformação que serão aprendidos dos dados.
scaler = MinMaxScaler()

# Combina duas etapas em uma:
# fit: Calcula os valores mínimo e máximo das colunas MedInc e AveOccup para determinar a escala.
# transform: Aplica a transformação aos dados, ajustando cada valor para o intervalo [0, 1]
# Os valores transformados substituem os valores originais nas colunas especificadas.
df[['MedInc', 'AveOccup']] = scaler.fit_transform(df[['MedInc', 'AveOccup']])

# Exibe as primeiras linhas do DataFrame após a transformação, mostrando os novos valores normalizados.
df[['MedInc', 'AveOccup']].head()

# Distribução dos dados após a normalização
# Compare esse gráfico com o primeiro, note como os valores dos eixos x e y
# agora estão entre 0 e 1
df.plot.scatter('MedInc', 'AveOccup')

# Z-score Standardization
# De forma análoga ao que foi realizado com o `Min-max Scaling`, agora será aplicado outra forma de normalização, o `Z-score Standardization`.
# `StandardScaler`: Centraliza os dados (média = 0) e escala para ter variância unitária.

# Cria um objeto da classe StandardScaler.
# Esse objeto armazenará os parâmetros de transformação que serão aprendidos dos dados.
scaler = StandardScaler()

# Combina duas etapas em uma:
# fit: Calcula os valores mínimo e máximo das colunas MedInc e AveOccup para determinar a escala.
# transform: Aplica a transformação aos dados, ajustando cada valor para que ao final
# o conjunto possua a média 0 e variância unitária.
# Os valores transformados substituem os valores originais nas colunas especificadas.
df[['MedInc', 'AveOccup']] = scaler.fit_transform(df[['MedInc', 'AveOccup']])

# Exibe as primeiras linhas do DataFrame após a transformação, mostrando os novos valores normalizados.
df[['MedInc', 'AveOccup']].head()

# Imprime os valores da média
# Nota-se que os valores obtidos são muito proximo de 0
print("Média dos valores após a normalização: ")
print(df[['MedInc', 'AveOccup']].mean())

# Imprime os valores do desvio padrão
# Nota-se que os valores obtidos são muito proximo de 1
print("Desvio padrão dos valores após a normalização: ")
print(df[['MedInc', 'AveOccup']].std())

# Distribução dos dados antes da normalização
# Compare esse gráfico com o primeiro e o do min max scaling,
# Note como os valores dos eixos x e y não estão entre 0 e 1
# nem estão nos valores originais, porém, como visto anteriormente
# mantém média 0 e desvio padrão 1
df.plot.scatter('MedInc', 'AveOccup')

# Codificação de Variáveis Categóricas
# Variáveis categóricas são aquelas que contêm valores discretos e não numéricos. Para usar essas variáveis em algoritmos de *machine learning*, precisamos convertê-las em uma forma numérica.
# Métodos de Codificação
# - **One-Hot Encoding:** Cria colunas binárias para cada categoria.
# - **Label Encoding:** Atribui um número inteiro único a cada categoria.

# One-Hot Encoding
# Cria colunas binárias para cada categoria.
# Serão utilizados os mesmos dados da etapa de normalização, por isso é importante não pular as etapas anteriores.

# Vizualização dos dados
df.head()

# Como abordado no ebook, muitos os algoritmos necessitam que colunas categóricas possuam números ao invés de textos.
# No trecho abaixo do código o objetivo é transformar os rótulos a coluna ocena_proximity  em números. Relembrando, esses rótulos foram criados artificialmente no começo da etapa de pré procesamento com rótulos do formato `string` podendo assumir valores: <1H OCEAN, INLAND ou NEAR OCEAN. No entanto, em alguns casos reais esses rótulos já podem vir nativamente no dataset.
# Abaixo será visto:
# - Criação de um one-hot encoder.
# - Aplicação do one-hot encoding à coluna ocean_proximity.
# - A agregação do DataFrame codificado ao original.
# - Remoção da coluna original codificada.
# - Exibição das primeiras linhas do DataFrame resultante.
# O resultado final é um DataFrame onde a coluna categórica ocean_proximity foi substituída por novas colunas binárias, representando cada categoria possível.

# Cria um codificador one-hot
encoder = OneHotEncoder(sparse_output=False)

# Aplica o one-hot encoding à coluna 'ocean_proximity'
#   - fit_transform: Ajusta o encoder aos dados e realiza a transformação de uma só vez
encoded_features = encoder.fit_transform(df[['ocean_proximity']])

# Cria um novo DataFrame com as features codificadas
#   - get_feature_names_out: Obtém os nomes das novas colunas geradas
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['ocean_proximity']))

# Combina o DataFrame original com o DataFrame codificado
#   - drop: Remove a coluna original 'ocean_proximity'
df_onehot = df.join(encoded_df).drop(columns=['ocean_proximity'])

# Visualiza as primeiras linhas do DataFrame com as features codificadas
# Note as últimas 3 colunas e os valores que elas assumem (0 ou 1)
df_onehot.head()

# Label Enconding
# De forma análoga ao visto com o `OneHotEncoder` aqui é utilizado o `LabelEncoder`, que atribui um número inteiro único a cada categoria.
# Cada categoria da coluna ocean_proximity receberá um número correspondente. Esse número substituirá o rótulo da coluna ocean_proximity, retirando permitindo que os algoritmos de machine learning possam funcionar adequadamente.

# Cria um codificador LabelEncoder
encoder = LabelEncoder()

# Aplica o LabelEncoder à coluna 'ocean_proximity', substituindo os valores existentes
#   - fit_transform: Ajusta o encoder aos dados e realiza a transformação de uma só vez
df['ocean_proximity'] = encoder.fit_transform(df['ocean_proximity'])

# Visualiza as primeiras linhas do DataFrame com as features codificadas
# Note as última  colunas e os valores que elas assumem (0 a 2)
df.head()

# Tabela das classes criadas
# Com base nela é possível decodificar a coluna ocean_proximity
encoder.classes_

# Exercício 1: Normalização
# Normalize a coluna `AveRooms` utilizando o Min-Max Scaling.

# Normalizar a coluna AveRooms utilizando Min-Max Scaling
scaler = MinMaxScaler()
df['AveRooms'] = scaler.fit_transform(df[['AveRooms']])
print(df[['AveRooms']].head())

# Conjuntos de Treino, Validação e Teste

# Objetivos
# - **Compreender a importância de dividir os dados em conjuntos de treino, validação e teste.**
# - **Aprender a dividir um conjunto de dados utilizando bibliotecas populares do Python.**
# - **Implementar e avaliar modelos de machine learning utilizando esses conjuntos de dados.**

# 1. Introdução
# Dividir os dados em conjuntos de treino, validação e teste é uma prática essencial em *machine learning*. Essa divisão permite avaliar o desempenho do modelo de forma objetiva e evitar problemas como o *overfitting*.

# Importância da Divisão dos Dados
# - **Treino:** Usado para ajustar os parâmetros do modelo.
# - **Validação:** Usado para ajustar os hiperparâmetros do modelo e selecionar o melhor modelo.
# - **Teste:** Usado para avaliar o desempenho final do modelo.

# Exemplo de aplicação da divisão dos dados
# Nesse exemplo utilizaremos o dataset do Titanic. O dataset do Titanic é um conjunto de dados bastante popular na comunidade de ciência de dados, especialmente para aqueles que estão iniciando seus estudos em machine learning. Ele contém informações sobre os passageiros do Titanic, incluindo:
# - **Informações demográficas**: Idade, sexo, classe social.
# - **Informações sobre a viagem**: Porto de embarque, número de familiares a bordo.
# - **Resultado**: Se o passageiro sobreviveu ou não ao naufrágio.
# Nesse parte, o objetivo é dividir esse dataset em treino, validação e teste utilizando duas técnicas distintas Holdout e K-fold.
# Importação dos dados

# Importa-se as bibliotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Carregar p conjunto de dados do Titanic para trabalhar como exemplo
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv'
df = pd.read_csv(url)

# Visualizar os primeiros dados
df.head()

# Holdout
# Aqui será apresentado a divisão dos dados em Treino, Validação e Teste. E o treino de um modelo simples para acompanhar a importância dessa divisão.
# Algumas etapas importante são:
# - **Seleciona as features relevantes**: Idade (Age) e tarifa (fare), como tentativa para predizer se o passegeiro sobreviveu o não
# - **Remove dados faltantes**: Garante que o modelo seja treinado com dados completos.
# - **Divide os dados**: Cria conjuntos de treinamento (60%), validação (20%) e teste (20%) para avaliar o desempenho do modelo e evitar overfitting.
# - **Verifica as dimensões**: Confirma que os conjuntos foram divididos corretamente.

# Definir features e target

# Remove as linhas com valores faltantes (NaN) das colunas 'age', 'fare', 'survived' utilizando o método dropna()
df_model = df[['age', 'fare', 'survived']].dropna()
# As colunas 'age' e 'fare' serão utilizadas como features (variáveis explicativas) para o modelo.
X = df_model[['age', 'fare']]
# A coluna 'survived' será o target (variável a ser prevista), indicando se o passageiro sobreviveu ou não.
y = df_model['survived']

# Dividir os dados em treino (60%), validação (20%) e teste (20%)
# O random_state é importante para garantir a reprodutibilidade dos resultados,
# especialmente quando se deseja comparar diferentes modelos ou realizar experimentos.
# Essa primeira chamada divide os dados em 60% para treinamento e 40% para validação e teste.
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)

# Essa segunda chamada divide os 40% restantes em partes iguais para validação e teste.
# O parâmetro random_state=42 garante que a divisão seja sempre a mesma, permitindo reprodutibilidade dos resultados.
# X_train, X_val, X_test: Contêm as features (idade e tarifa) para os conjuntos de treinamento, validação e teste, respectivamente.
# y_train, y_val, y_test: Contêm os valores correspondentes da variável target (sobreviveu ou não) para os conjuntos de treinamento, validação e teste, respectivamente.
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Impressão do tamanhos dos conjuntos de dados
# Note que os tamanhos obedecem 60% para treino, 20% para validdação e 20% para teste
print(f"Tamanho do conjunto de treino: {X_train.shape}")
print(f"Tamanho do conjunto de validação: {X_val.shape}")
print(f"Tamanho do conjunto de teste: {X_test.shape}")

# Treinando o Modelo
# O código apresentado define uma função `treina_modelo` que encapsula o processo de treinamento e avaliação de um modelo de regressão logística.
# A função `treina_modelo` recebe como entrada os dados de treinamento e validação, treina um modelo de regressão logística e avalia seu desempenho nos conjuntos de validação e teste, fornecendo como saída as respectivas acurácias.
# **Entradas**:
# - **X_train**: Um DataFrame contendo as features (nesse caso, 'age' e 'fare') dos passageiros do conjunto de treinamento. Essas features serão utilizadas pelo modelo para aprender a prever se um passageiro sobreviveria ao naufrágio do Titanic.
# - **y_train**: Um Series contendo os valores da variável target (sobreviveu ou não) correspondentes aos dados de treinamento.
# - **X_val**: Um DataFrame contendo as features do conjunto de validação, utilizado para ajustar os hiperparâmetros do modelo e avaliar seu desempenho em dados não vistos durante o treinamento.
# - **y_val**: Um Series contendo os valores da variável target correspondentes aos dados de validação.
# **Saída**:
# A função não retorna nenhum valor explicitamente. Em vez disso, ela imprime na tela duas métricas de desempenho:
# - **Acurácia no conjunto de validação**: Indica a proporção de exemplos do conjunto de validação que foram classificados corretamente pelo modelo.
# - **Acurácia no conjunto de teste**: Indica a proporção de exemplos do conjunto de teste que foram classificados corretamente pelo modelo. Essa métrica fornece uma estimativa mais confiável do desempenho do modelo em dados nunca vistos antes.

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def treina_modelo(X_train, y_train, X_val, y_val):
    # Treinar o modelo no conjunto de treino
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Avaliar o modelo no conjunto de validação
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Acurácia no conjunto de validação: {val_accuracy:.2f}")

    # Avaliar o modelo no conjunto de teste
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Acurácia no conjunto de teste: {test_accuracy:.2f}")

treina_modelo(X_train, y_train, X_val, y_val)

# Validação Cruzada: K-Fold Cross-Validation
# Validação cruzada é vantajosa porque pode retira o fator "sorte"  e de ter realizado uma divisão vantajosa para o modelo.
# No K-fold dividimos os dados de treino em `k` grupos e comparamos entre eles.
# Abaixo é a aplicação da validação cruzada k-fold (com k=5) para avaliar o desempenho do modelo de regressão logística.
# **Entradas**:
# - **X_train**: Um DataFrame contendo as features (nesse caso, 'age' e 'fare') do conjunto de treinamento completo.
# - **y_train**: Um Series contendo os valores da variável target (sobreviveu ou não) correspondentes aos dados de treinamento completo.
# - **kf**: Um objeto KFold que define a estratégia de divisão dos dados em 5 folds.
# **Saída**:
# A cada iteração do loop, o código:
# - **Divide os dados**: Separa os dados de treinamento em um conjunto de treinamento e um conjunto de validação, de acordo com os índices definidos pelo objeto KFold.
# - **Treina o modelo**: Chama a função treina_modelo com os conjuntos de treinamento e validação atuais, treinando o modelo de regressão logística.
# - **Avalia o modelo**: A função treina_modelo calcula e imprime a acurácia do modelo nos conjuntos de validação e teste.
# O objetivo principal da validação cruzada k-fold é obter uma estimativa mais robusta e confiável do desempenho do modelo, reduzindo o viés que poderia ocorrer devido a uma única divisão aleatória dos dados em treinamento e teste. Ao repetir o processo de treinamento e validação 5 vezes, utilizando diferentes combinações dos dados como conjunto de treinamento e validação, obtém-se diferentes métricas de desempenho, o que proporciona uma visão mais precisa do desempenho geral do modelo.

# Importa a classe KFold da biblioteca sklearn
from sklearn.model_selection import KFold

# Define o número de 5 folds para a validação cruzada
kf = KFold(n_splits=5)

# Itera sobre cada fold
for train_index, val_index in kf.split(X_train):
    # Divide os dados em conjuntos de treino e validação para o fold atual
    X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

    # Aqui chama-se a função de treinamento anteriormente definida
    # verifica-se as métricas a cada fold
    treina_modelo(X_train_fold, y_train_fold, X_val_fold, y_val_fold)

# Exercício 1
# Divida o conjunto de dados do titanic em conjuntos de treino (70%), validação (15%) e teste (15%).
# OBS. Para questões de comparação, utlize o `random_state` igual a 42


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Carregar um conjunto de dados de exemplo
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv'
df = pd.read_csv(url)

# Visualizar os primeiros dados
df.head()

# Dividir os dados
# O parâmetro random_state=42 garante que a divisão seja sempre a mesma, permitindo reprodutibilidade dos resultados.
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Tamanho do conjunto de treino: {X_train.shape}")
print(f"Tamanho do conjunto de validação: {X_val.shape}")
print(f"Tamanho do conjunto de teste: {X_test.shape}")

# Dados balanceados e desbalanceados

# Objetivos
# - **Compreender a importância de dados balanceados em modelos de machine learning.**
# - **Identificar técnicas para lidar com dados desbalanceados.**
# - **Implementar essas técnicas utilizando bibliotecas populares do Python.**

# 1. Introdução
# Em muitos problemas de *machine learning*, especialmente aqueles envolvendo classificação, os dados podem estar desbalanceados, significando que algumas classes têm muito mais exemplos do que outras. Isso pode afetar o desempenho do modelo, levando a previsões tendenciosas.

# 2. Explicação Teórica

# Dados Balanceados vs. Desbalanceados
# - **Dados Balanceados:** Quando as classes no conjunto de dados têm aproximadamente o mesmo número de exemplos.
# - **Dados Desbalanceados:** Quando uma ou mais classes têm significativamente mais exemplos do que outras.

# 3. Exemplos Práticos
# A exemplo do que foi utilizado anteriormente, o dataset do Titanic agora será explorado as questões de desbalanceamento.
# Teremos algumas etapas a serem seguidas:
# - **Seleção das colunas relevantes para a análise de sobrevivência**: 'pclass' (classe social), 'age' (idade), 'sibsp' (número de irmãos/cônjuges a bordo) e 'fare' (tarifa paga).
# - **Remoção das linhas com valores ausentes (NaN)** para evitar erros no processamento posterior.
# - **Divisão em features e target**: Separa o DataFrame em features (X) contendo as colunas selecionadas e a target (y) contendo a informação de sobrevivência (0 - não sobreviveu, 1 - sobreviveu).
# - **Divisão treino e teste**: Divide os dados em conjuntos de treinamento (X_train, y_train) e teste (X_test, y_test) utilizando a função train_test_split com uma proporção de 70% para treinamento e 30% para teste. Define um random_state para garantir reprodutibilidade dos resultados.
# - **Visualização da distribuição da classe target**: Imprime a contagem de valores da variável target no conjunto de treinamento, permitindo verificar a distribuição entre as classes "sobreviveu" e "não sobreviveu" (valor idealmente balanceado para um modelo de classificação).

# Carregando Bibliotecas e Dados

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Carregar um conjunto de dados de exemplo
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv'
df = pd.read_csv(url)

# Filtrar colunas relevantes e remover valores ausentes
df = df[['survived', 'pclass', 'age', 'sibsp', 'fare']].dropna()

# Dividir features e target
X = df.drop(columns=['survived'])
y = df['survived']

# Dividir os dados em treino (70%) e teste (30%)
# O parâmetro random_state=42 garante que a divisão seja sempre a mesma, permitindo reprodutibilidade dos resultados.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Visualizar distribuição da classe target
print(y_train.value_counts())

# Treina o modelo
# Essa é a mesma função apresentada no item de Conjuntos de Treino, Validação e Teste. Caso tenha dúvida sobre seu funcionamento, volte na célula https://colab.research.google.com/drive/1s0wK-XDQOdUubutjqhJVQL-6q6jKAfy9#scrollTo=urlttD4B61SW

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def treina_modelo(X_train, y_train, X_val, y_val):
    # Treinar o modelo no conjunto de treino
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Avaliar o modelo no conjunto de validação
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Acurácia no conjunto de validação: {val_accuracy:.2f}")

    # Avaliar o modelo no conjunto de teste
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Acurácia no conjunto de teste: {test_accuracy:.2f}")

# Oversampling com SMOTE
# SMOTE cria dados artificiais com base nos `K` vizinhos mais póximos.
# Algoritmo simplificado:
# 1. Seleciona a classe minoritária
# 2. Utiliza o KNN na classe minoritária com algum valor de `K`
# 3. Desenha-se linhas entre uma amostra da classe minoritária e seus `K` vizinhos
# 4. Escolhe um valor aleatório sobre essa linha
# <figure style="background-color: white; padding: 10px;">
# <img src="https://i0.wp.com/varshasaini.in/wp-content/uploads/2022/07/smote.jpeg?w=1400&ssl=1" style="display: block; margin: 0 auto;">
# <figcaption align="center">FONTE: https://i0.wp.com/varshasaini.in/wp-content/uploads/2022/07/smote.jpeg?w=1400&ssl=1</figcaption>
# </figure>
# A seguir é apresentado a implementação do oversampling com o SMOTE. Como entrada tem-se os dados do titanic anteriormente separados. Como saída tem-se ambas classes com a mesma quantidade (da classe com maior número de elementos)
# Compare a quantidade de amostras `y_train_smote` apresentado abaixo com o `y_train` apresentado anteriormente. Note que a classe 0 é a que possui mais exemplos (298), enquanto que a classe 1 que possuia 201 antes, agora possui 298.

# cria o modelo smote através da classe SMOTE.
# novamente o random_state é utilizado para que os dados sejam replicáveis
smote = SMOTE(random_state=42)
# Cria novos dados com base no X_train e y_train utilizando o SMOTE.
# Esses novos dados são guardados nas variáveis X_train_smote e y_train_smote
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Visualizar distribuição da classe target após SMOTE
print(y_train_smote.value_counts())

# Compare as métricas de acurácia antes e depois do balanceamento.
# Nota-se uma melhor acurácia dos dados quando há o balanceamento de classes.

# Comparação da acurácia com o balancelamento

print("Modelo treinado ignorando o desbalancemanento")
treina_modelo(X_train, y_train, X_test, y_test)

print("\n************************************\n")
print("Modelo treinado após o SMOTE")
treina_modelo(X_train_smote, y_train_smote, X_test, y_test)

# Undersampling
# Diminui o dataset, retirando aleatóriamente da classe majoritária alguns exemplos
# Nessa parte será apresentado o `RandomUnderSampler`que remove aleatoriamente dados da classe majoritária (classe 0).
# Apesar de trazer o mesmo efeito de balanceamento entre classes, no undersampling reduz-se a quantidade de amostras. Nesse caso, serão eliminados aleatoriamente dados dos sobrevivente (rotulados com 1 na survived).

# cria o modelo undersampler através da classe RandomUnderSampler.
# novamente o random_state é utilizado para que os dados sejam replicáveis
undersampler = RandomUnderSampler(random_state=0)
# Eliminam-se dados com base no X_train e y_train utilizando o RandomUnderSampler.
# Os novos dados são guardados nas variáveis X_train_under e y_train_under
X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

# Visualizar distribuição da classe target após undersampling
print(y_train_under.value_counts())

# Exercício 1: Aplicar *Undersampling* e Treinar Modelo
# Utilize a função `treina_modelo` (criada anteriormente) com os dados X_train, y_train, X_test e y_test também criados anteriormente.
# Utilize a função `treina_modelo` (criada anteriormente) com os dados X_train_under, y_train_under, X_test e y_test também criados anteriormente.
# Compare os resultados de acurácia do modelo treinado original (dados desbalanceados) e com dados balanceados pela técnica de undersampling.

# Exercício Resolvido
# Comparação da acurácia com o balancelamento

print("Modelo treinado ignorando o desbalancemanento")
treina_modelo(X_train, y_train, X_test, y_test)

print("\n************************************\n")
print("Modelo treinado após o Undersampling")
treina_modelo(X_train_under, y_train_under, X_test, y_test)

# *Underfitting* e *Overfitting*

# Objetivos
# - **Compreender os conceitos de *underfitting* e *overfitting* em modelos de *machine learning*.**
# - **Aprender a identificar *underfitting* e *overfitting* através de métricas de desempenho e visualizações.**
# - **Implementar técnicas para prevenir *underfitting* e *overfitting* em modelos utilizando Python.**

# 1. Introdução
# Em machine learning, underfitting e overfitting são problemas comuns que afetam a capacidade de generalização dos modelos. Underfitting ocorre quando um modelo é muito simples para capturar a complexidade dos dados, enquanto overfitting ocorre quando um modelo é muito complexo e captura o ruído dos dados de treino. Este notebook abordará esses conceitos e apresentará técnicas para mitigá-los.

# 3. Exemplos Práticos
# Exemplo 1
# Nesse primeiro exemplo servirá para ilustrar modelos com underfitting, overfitting e um modelo ideal para o mesmo conjunto de dados.
# - O modelo real é definido pela função `true_function` e é uma função quadrática $x^2$.
# - Utiliza-se essa função para gerar dados sintéticos e aleatórios ao redor desse modelo ideal, representado pelos pontos em vermelho e as variáveis `x` e `y`. `x` e `y` podem ser considerados o conjunto de dados de treino.
# - Criam-se os dados de teste `x_fit` e `y_true`
# Carregando Bibliotecas e Dados

# Importação das bibliotecas que serão utilizadas
import matplotlib.pyplot as plt
import numpy as np

# Define o modelo ideal como sendo uma função quadrática
def true_function(x):
    return x**2

# Gerar dados sintéticos
# Define-se um gerador aleatório que seja reprodutível, é semelhante ao parÂmetro random_state visto anteriormente
np.random.seed(22)

# Geram-se 10 valores aleatórios entre -2 e 2
x = np.sort(np.random.rand(10) * 4 - 2)  # random values between -2 and 2

# Para cada valor gerado anteriormente, aplica a função verdade `true_function(x)` para obter o valor de y
# adiciona-se um ruído com `np.random.randn(10)` para que o valor não caia sempre sobre o próprio modelo
y = true_function(x) + np.random.randn(10)

# Criam-se os dados de teste
# x_fit são 100 valores igualmente espaçados entre -2 e 2
x_fit = np.linspace(-2, 2, 100)
# y_true é o resultado do modelo ideal para os dados de x_fit
y_true = true_function(x_fit)

# Underfitting
# Para exemplificar o underfitting é utilizado o modelo de regressão linear `model_underfit`.
# Os dados de treino `x` e `y` servem para treinar o modelo linear.
# Com o modelo treinado testa-se seu desempenho passando os valores de teste `x_fit`, obtendo as predições e guardando em `y_underfit`.
# Esses valores de `y_underfit` são utilizados em conjunto com `x_fit` para obter a curva do modelo ajustado da figura a esquerda.

from sklearn.linear_model import LinearRegression

# Cria o modelo de regressão linear
model_underfit = LinearRegression()
# Treina o modelo com dados de treino
model_underfit.fit(x.reshape(-1, 1), y)
# Obtem-se os resultados do modelo através do teste dos valores de x_fit
y_underfit = model_underfit.predict(x_fit.reshape(-1, 1))

# Overfitting
# Para exemplificar o overfitting é utilizado o modelo de regressão polinomial de grau 4 definido por `model_overfit`. O modelo de regressão polinomial é um modelo mais complexo que o de regressão linear utilizado no underfitting.
# O pipeline consiste em dois passos:
# - **Passo 1 (`PolynomialFeatures(degree)`)**: Primeiro, o pipeline aplica a transformação PolynomialFeatures com o grau definido anteriormente (4). Essa transformação gera novas features polinomiais a partir da feature original x.
# - **Passo 2 (`LinearRegression()`)**: Em seguida, o pipeline aplica um modelo de regressão linear para aprender a relação entre as features polinomiais geradas na etapa anterior e a variável target y.
# Os dados de treino `x` e `y` servem para treinar o modelo.
# Com o modelo treinado testa-se seu desempenho passando os valores de teste `x_fit`, obtendo as predições e guardando em `y_overfit`.
# Esses valores de `y_overfit` são utilizados em conjunto com `x_fit` para obter a curva do modelo ajustado da figura a direita.

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Define o grau do modelo polinomial
degree = 4

# Cria o modelo de regresão polinomial
model_overfit = make_pipeline(PolynomialFeatures(degree), LinearRegression())
# Treina o modelo com dados de treino
model_overfit.fit(x.reshape(-1, 1), y)
# Obtem-se os resultados do modelo através do teste dos valores de x_fit
y_overfit = model_overfit.predict(x_fit.reshape(-1, 1))

# Comparação
# No código abaixo são gerados 2 gráficos de comparação, um para o underfitting e outro para o overfitting.
# Em cada gráfico são plotados:
# - Os dados de teste `x` e `y` representados por pontos vermelhos.
# - A curva fruto do modelo real `true_function` fruto do resultado da aplicação do conjunto de teste `x_fit` que obteve `y_true`. A vurva é representada pela linha pontilhada.
# - A curva fruto do modelo (underfitting ou overfitting) fruto do resultado da aplicação do conjunto de teste `x_fit` no modelo `model_overfit` ou `model_underfit` que obteveram `y_overfit` e `y_underfit` respectivamente. Essas curvas são representadas pelas linhas cheias.

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Underfitting plot
axs[0].scatter(x, y, color='red', label='Pontos do dataset')
axs[0].plot(x_fit, y_true, 'k--', label='Curva Verdadeira')
axs[0].plot(x_fit, y_underfit, 'k-', label='Curva Ajustada')
axs[0].set_title('Underfitting')
axs[0].legend()

# Overfitting plot
axs[1].scatter(x, y, color='red', label='Pontos do dataset')
axs[1].plot(x_fit, y_true, 'k--', label='Curva Verdadeira')
axs[1].plot(x_fit, y_overfit, 'k-', label='Curva Ajustada')
axs[1].set_title('Overfitting')
axs[1].legend()

plt.show()

# Exemplo 2
# Mais um exemplo agora para observar o impacto da escolha do parâmetro do modelo (degree) nas métricas de performance.
# O código gera dados sintéticos com relação não linear, divide os dados em treino e teste, ordena os dados de teste para plotagem e define duas funções: uma para plotar os resultados do modelo e outra para calcular e imprimir métricas de desempenho (MSE e $R^2$). Este código base pode ser utilizado para treinar diferentes modelos de regressão com features polinomiais e avaliar o seu desempenho.
# O código pode ser dividio em algumas partes:
# Geração de Dados Sintéticos:
# A exemplo do que foi abordado no exemplo anterior, essa fase visa criar alguns dados fictícios.
# - `np.random.seed(0)`: Define uma semente para geração de números aleatórios, garantindo reprodutibilidade dos resultados.
# - `X = np.sort(5 * np.random.rand(80, 1), axis=0)`: Gera um array X com 80 linhas e 1 coluna, contendo valores aleatórios entre 0 e 5, ordenados em ordem crescente.
# - `y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])`: Gera a variável target y baseada no seno dos valores de X com adição de um ruído gaussiano (normal) de média 0 e desvio padrão 0.1. A função ravel() transforma o array y em unidimensional.
# Divisão de Dados:
# Como visto anteriormente, a divisão dos dados em treino e teste é uma fase importante para a avaliação dos modelos.
# - **train_test_split**: Divide X e y em conjuntos de treinamento (X_train, y_train) e teste (X_test, y_test) com uma proporção de 80% para treinamento e 20% para teste. random_state=0 garante reprodutibilidade.
# Ordenação para Plotagem:
# Uma fase para auxiliar na visualização dos dados.
# - **z = sorted(zip(X_test,y_test))**: Cria uma lista z ordenando os pares (valor de X, valor de y) do conjunto de teste.
# - **X_test,y_test = zip(*z)**: Desempacota a lista ordenada z separando novamente as listas de X_test e y_test. Isso garante que os pontos de dados no teste sejam plotados na mesma ordem da previsão do modelo.
# Função de Plotagem:
# Para evitar a repetição de códigos, foi definida uma função específica para a plotagem dos gráficos.
# **plot_results**: Função que recebe os dados de treinamento e teste, o modelo, features polinomiais (se houver) e um título, e gera um gráfico:
# - Plota os dados de treinamento (azul) e teste (vermelho) como pontos dispersos.
# - Plota a linha de previsão do modelo (verde) utilizando os dados de teste e as features polinomiais (se houver).
# - Adiciona título e legenda ao gráfico.
# - Exibe o gráfico utilizando plt.show().
# Função de Métricas:
# É definida uma função para obtenção das métricas cujo objetivo é análogo à função de plotagem.
# **print_metrics**: Função que recebe os valores verdadeiros (y_true), os valores previstos (y_pred) e o nome do modelo, e calcula e imprime as métricas:
# - **mse**: Calcula o erro médio quadrático (Mean Squared Error) entre os valores verdadeiros e previstos.
# - **r2**: Calcula o R-quadrado ($R^2$), que indica a proporção da variância explicada pelo modelo.
# - Imprime o nome do modelo seguido das métricas MSE e R^2 formatadas com 4 casas decimais.
# Carregando Bibliotecas e Dados

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# Gerar dados sintéticos
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])  # Adiciona ruído

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Ordena os dados para facilitar a plotagem
z = sorted(zip(X_test,y_test))
X_test,y_test = zip(*z)

# Função para plotar os resultados
def plot_results(X_train, y_train, X_test, y_test, model, X_poly, title):
    plt.scatter(X_train, y_train, color='blue', label='Dados de Treinamento')
    plt.scatter(X_test, y_test, color='red', label='Dados de Teste')
    plt.plot(X_test, model.predict(X_poly), color='green', label='Modelo')
    plt.title(title)
    plt.legend()
    plt.show()

# Função para calcular e imprimir as métricas
def print_metrics(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - MSE: {mse:.4f}, R^2: {r2:.4f}")

# Underfitting: Modelo Linear Simples
# O código apresentado simula um caso de underfitting utilizando um modelo de regressão linear com features polinomiais de grau 1. Vamos analisar passo a passo:
# **Criando Features Polinomiais para Underfitting**:
# - `poly_features_underfit = PolynomialFeatures(degree=1)`: Cria uma instância da classe PolynomialFeatures definindo o grau como 1.
# - `X_poly_train_underfit = poly_features_underfit.fit_transform(X_train)`: Aplica o transformador poly_features_underfit aos dados de treinamento X_train. A função fit_transform encaixa o transformador nos dados de treinamento (calcula parâmetros internos, se necessário) e então aplica a transformação.
# - `X_poly_test_underfit = poly_features_underfit.transform(X_test)`: Aplica o transformador poly_features_underfit (já encaixado nos dados de treinamento) aos dados de teste X_test.
# **Treinando o Modelo de Underfitting**:
# - `poly_reg_underfit = LinearRegression()`: Cria um modelo de regressão linear.
# - `poly_reg_underfit.fit(X_poly_train_underfit, y_train)`: Treina o modelo poly_reg_underfit utilizando as features polinomiais transformadas do treino (X_poly_train_underfit) e a variável target y_train.
# **Plotando os Resultados e Imprimindo Métricas**:
# **plot_results**:
# Função definida anteriormente para plotar os resultados do modelo. Ela recebe os dados de treinamento e teste, o modelo treinado (poly_reg_underfit), as features polinomiais transformadas (X_poly_test_underfit) e um título descritivo ("Underfitting: Modelo Polinomial de Grau 1").
# A função irá plotar os dados de treinamento e teste, e a linha de previsão do modelo utilizando as features polinomiais transformadas do teste (X_poly_test_underfit).
# O título indica que este é um caso de underfitting devido ao uso de um polinômio de grau 1.
# **print_metrics**:
# Função definida anteriormente para calcular e imprimir métricas de desempenho (MSE e $R^2$).
# Ela é chamada duas vezes:
# 1. A primeira vez com os dados de treinamento (y_train) e a previsão do modelo no treinamento (poly_reg_underfit.predict(X_poly_train_underfit)) para avaliar o desempenho no conjunto de treino.
# 2. A segunda vez com os dados de teste (y_test) e a previsão do modelo no teste (poly_reg_underfit.predict(X_poly_test_underfit)) para avaliar o desempenho no conjunto de teste.

# Criando Features Polinomiais para Underfitting
poly_features_underfit = PolynomialFeatures(degree=1)  # Grau 1 para underfitting
X_poly_train_underfit = poly_features_underfit.fit_transform(X_train)
X_poly_test_underfit = poly_features_underfit.transform(X_test)

# Treinando o Modelo de Underfitting
poly_reg_underfit = LinearRegression()
poly_reg_underfit.fit(X_poly_train_underfit, y_train)
plot_results(X_train, y_train, X_test, y_test, poly_reg_underfit, X_poly_test_underfit, "Underfitting: Modelo Polinomial de Grau 1")

# Plotando os Resultados e Imprimindo Métricas
print_metrics(y_train, poly_reg_underfit.predict(X_poly_train_underfit), "Underfitting: Modelo Polinomial de Grau 1 - Treino")
print_metrics(y_test, poly_reg_underfit.predict(X_poly_test_underfit), "Underfitting: Modelo Polinomial de Grau 1 - Teste")

# Overfitting: Modelo Polinomial de Alto Grau
# Semelhante ao apresentado anteriormente, com a única diferença que agora elevou-se o grau do polinômio para 15 (degree=15) para obter mais features e elevar a complexidade do modelo.

# Criando Features Polinomiais para overfitting
poly_features_overfit = PolynomialFeatures(degree=15)  # Grau alto para overfitting
X_poly_train_overfit = poly_features_overfit.fit_transform(X_train)
X_poly_test_overfit = poly_features_overfit.transform(X_test)

# Treinando o Modelo de overfitting
poly_reg_overfit = LinearRegression()
poly_reg_overfit.fit(X_poly_train_overfit, y_train)
plot_results(X_train, y_train, X_test, y_test, poly_reg_overfit, X_poly_test_overfit, "Overfitting: Modelo Polinomial de Alto Grau")

# Plotando os Resultados e Imprimindo Métricas
print_metrics(y_train, poly_reg_overfit.predict(X_poly_train_overfit), "Overfitting: Modelo Polinomial de Alto Grau - Treino")
print_metrics(y_test, poly_reg_overfit.predict(X_poly_test_overfit), "Overfitting: Modelo Polinomial de Alto Grau - Teste")

# Fitting Adequado: Modelo Polinomial de Grau 3
# Semelhante ao apresentado anteriormente, com a única diferença que agora utilizou o grau do polinômio 3 (degree=3) para obter mais features e adequar a complexidade do modelo.

poly_features_fit = PolynomialFeatures(degree=3)  # Grau intermediário para fitting adequado
X_poly_train_fit = poly_features_fit.fit_transform(X_train)
X_poly_test_fit = poly_features_fit.transform(X_test)

poly_reg_fit = LinearRegression()
poly_reg_fit.fit(X_poly_train_fit, y_train)
plot_results(X_train, y_train, X_test, y_test, poly_reg_fit, X_poly_test_fit, "Fitting Adequado: Modelo Polinomial de Grau 3")

print_metrics(y_train, poly_reg_fit.predict(X_poly_train_fit), "Fitting Adequado: Modelo Polinomial de Grau 3 - Treino")
print_metrics(y_test, poly_reg_fit.predict(X_poly_test_fit), "Fitting Adequado: Modelo Polinomial de Grau 3 - Teste")

# **O que observar?**
# Observe as metricas MSE e $R^2$ entre os conjuntos de Treino e Teste em cada modelo.
# - **Underfitting**: valores elevado de MSE quando comparados com os modelos adequado e overfitting. Isso indica que os outros modelo representam melhor. Isso já era esperado uma vez que o modelo linear (grau 1) é o mais simples possível.
# - **Overfitting**: grande diferença no MSE quando comparado o dataset de Treino e Teste. Nesse caso, o modelo acaba se ajustando ao treino e é incapaz de generalizar para os novos dados (Teste).
# - **Adequado**: Não possui o melhor resultado de MSE no treino, mas obteve o melhor MSE no Teste. Também não há grandes diferenças de MSE entre o conjunto de Treino e Teste, indicando que o modelo conseguiu generalizar e manteve o mesmo acerto tanto para os dados que ele conhecia (Treino) como para os que ele nunca viu (Teste).

# Referências Bibliográficas
# - McKinney, W. (2017). **Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython.** O'Reilly Media.
# - VanderPlas, J. (2016). **Python Data Science Handbook: Essential Tools for Working with Data.** O'Reilly Media.
# - Seaborn Documentation: https://seaborn.pydata.org/
# - Scikit-learn Documentation: https://scikit-learn.org/stable/
# - Pandas Documentation: https://pandas.pydata.org/
# ----------------------------------------------------------------------
