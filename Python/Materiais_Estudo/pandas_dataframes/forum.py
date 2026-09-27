import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carregando o dataset Iris
iris = sns.load_dataset('iris')

# Gráfico de violino
plt.figure(figsize=(10, 6))

sns.violinplot(
    data=iris,
    x='species',
    y='petal_length'
)

plt.title('Distribuição do Comprimento da Pétala por Espécie')
plt.xlabel('Espécie')
plt.ylabel('Comprimento da pétala (cm)')

plt.show()

# Calculando a média do comprimento da pétala por espécie
media_petala = iris.groupby('species')['petal_length'].mean()

# Gráfico de barras
plt.figure(figsize=(10, 6))

sns.barplot(
    data=iris,
    x='species',
    y='petal_length',
    estimator='mean'
)

plt.title('Média do Comprimento da Pétala por Espécie')
plt.xlabel('Espécie')
plt.ylabel('Média do comprimento da pétala (cm)')

plt.show()

# Exibindo os valores das médias
print(media_petala)

# Boxplot do comprimento da pétala por espécie
plt.figure(figsize=(10, 6))

sns.boxplot(
    data=iris,
    x='species',
    y='petal_length'
)

plt.title('Boxplot do Comprimento da Pétala por Espécie')
plt.xlabel('Espécie')
plt.ylabel('Comprimento da pétala (cm)')

plt.show()