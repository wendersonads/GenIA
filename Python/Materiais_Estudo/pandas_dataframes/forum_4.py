import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv'

titanic = pd.read_csv(url)

#Exibir as primeiras linhas do dataset
print(titanic.head(), '\n')

#Exibir e listar todas as colunas e seus tipos
print(titanic.dtypes)

#Adicionando nova coluna
titanic['family_size'] = titanic['sibsp'] + titanic['parch']

#Conferindo nova coluna adicionada
print(titanic[['sibsp', 'parch', 'family_size']].head(10), '\n')

#Apresentando gráfico para verificar se o tamanho da família influenciou na taxa de sobrevivência
sns.violinplot(x='survived', y='family_size', data=titanic)
plt.title('Distribuição de familias entre sobreviventes')
plt.xlabel('Sobreviventes')
plt.ylabel('Familia')
plt.show()

#Apresentando gráfico para verificar relacão entre idade e tarica de acrdo com a sobrevivência
plt.figure(figsize=(10, 6))

sns.scatterplot(
    x='age',
    y='fare',
    hue='survived',
    data=titanic,
    alpha=0.6,
    palette={0: 'red', 1: 'green'}
)

plt.title('Relação entre Idade, Tarifa Paga e Sobrevivência')
plt.xlabel('Idade')
plt.ylabel('Tarifa Paga')
plt.legend(title='Sobreviveu', labels=['Não', 'Sim'])

plt.show()
