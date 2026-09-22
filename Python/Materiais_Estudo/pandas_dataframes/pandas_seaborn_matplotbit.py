import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregando o dataset do Titanic diretamente do GitHub
url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv'
titanic = pd.read_csv(url)

# Histograma da distribuição de idades
(titanic['age']
 .dropna()
 .plot(
    kind='hist',
    bins=30,
    color='c',
    title='Distribuição de Idades no Titanic'
    )
 )
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.show()

# Carrega o conjunto de dados Iris
iris = sns.load_dataset('iris')

# Define o tamanho da figura
plt.figure(figsize=(10, 6))

plt.plot(iris['sepal_length'], label='Comprimento da Sépala')
plt.plot(iris['sepal_width'], label='Largura da Sépala')

plt.title('Medidas das Sépalas das Íris')
plt.xlabel('Índice')
plt.ylabel('Comprimento em cm')
plt.legend()
plt.show()

#Agrupando os dados por espécie e calculando a média do comprimento das sépalas
sepal_length_means = iris.groupby('species')['sepal_length'].mean()

# Criando um gráfico de barras
sepal_length_means.plot(kind='bar', color='cyan', alpha=0.7)

plt.title('Média do Comprimento das Sépalas por Espécie')
plt.xlabel('Espécie')
plt.ylabel('Média do Comprimento das Sépalas (cm)')
plt.show()

# Define a cor, a transparência (alpha) e o número de barras (bins)
plt.hist(iris['petal_length'], bins=20, color='green', alpha=0.7)

plt.title('Distribuição do Comprimento das Pétalas')
plt.xlabel('Comprimento das Pétalas em cm')
plt.ylabel('Frequência')
plt.show()

# Preparando os dados para o gráfico de área
x = range(len(iris))  # Índice para cada amostra

# Comprimento e largura da sépala, tratando possíveis dados faltantes
y_sepal_length = iris['sepal_length'].fillna(0)
y_sepal_width = iris['sepal_width'].fillna(0)

# Criando um gráfico de área
plt.figure(figsize=(10, 6))
plt.stackplot(x, y_sepal_length, y_sepal_width, labels=['Comprimento da Sépala', 'Largura da Sépala'], colors=['lightblue', 'lightgreen'], alpha=0.5)
plt.title('Distribuição Acumulada das Medidas das Sépalas')
plt.xlabel('Índice da Amostra')
plt.ylabel('Medidas das Sépalas (cm)')
plt.legend(loc='upper left')
plt.show()


plt.scatter(iris['petal_length'], iris['petal_width'], c='red', alpha=0.5)
plt.title('Gráfico de Dispersão do Comprimento vs Largura das Pétalas')
plt.xlabel('Comprimento das Pétalas em cm')
plt.ylabel('Largura das Pétalas em cm')
plt.show()

# Contando quantas amostras existem de cada espécie
species_counts = iris['species'].value_counts()
# Criando um gráfico de pizza
plt.figure(figsize=(8, 8))
plt.pie(species_counts, labels=species_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen', 'lavender'])
plt.title('Distribuição das Espécies de Íris')
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(14, 7))
axs[0].plot(iris['sepal_length'], iris['sepal_width'], 'b^')
axs[1].hist(iris['sepal_width'], bins=15, color='r', alpha=0.7)
plt.tight_layout()
plt.show()

sns.pairplot(iris, hue='species', markers=["o", "s", "D"])
plt.suptitle('Pair Plot das Variáveis do Dataset Iris', y=1.02)
plt.show()

sns.jointplot(x='petal_length', y='petal_width', data=iris, kind='scatter', color='m')
plt.suptitle('Joint Plot de Comprimento e Largura das Pétalas', y=1.02)
plt.show()

plt.figure(figsize=(8, 6))
sns.kdeplot(iris['sepal_length'], fill=True, color="r", label="Comprimento da Sépala")
sns.kdeplot(iris['sepal_width'], fill=True, color="b", label="Largura da Sépala")
plt.title('KDE Plots de Comprimento e Largura das Sépalas')
plt.xlabel('Centímetros')
plt.ylabel('Densidade')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='species', y='petal_length', data=iris)
plt.title('Distribuição do Comprimento das Pétalas por Espécie')
plt.xlabel('Espécie')
plt.ylabel('Comprimento das Pétalas (cm)')
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x='species', y='sepal_width', data=iris, palette='Pastel1', hue='species', legend=False)
plt.title('Distribuição da Largura das Sépalas por Espécie')
plt.xlabel('Espécie')
plt.ylabel('Largura das Sépalas (cm)')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='species', y='sepal_length', data=iris, palette='Accent', hue='species', legend=False)
plt.title('Média do Comprimento das Sépalas por Espécie')
plt.xlabel('Espécie')
plt.ylabel('Comprimento Médio das Sépalas (cm)')
plt.show()

# Definindo o estilo do gráfico
sns.set_style("whitegrid")
# Personalizando a paleta de cores
sns.set_palette("husl")

# Definição do Gŕafico ...