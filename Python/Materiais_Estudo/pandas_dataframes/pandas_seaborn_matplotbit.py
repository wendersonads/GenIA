import pandas as pd
import matplotlib.pyplot as plt

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