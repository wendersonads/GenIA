import pandas as pd
import numpy as np

s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s, '\n')

data = {
    'Nome': ['João', 'Ana', 'Pedro', 'Maria'],
    'Idade': [28, 22, 35, 45],
    'Cidade': ['Goiânia', 'São Paulo', 'Salvador', 'Curitiba']
}

df = pd.DataFrame(data)
print(df, '\n')

df_csv = pd.read_csv('../../../dados.csv')
print(df_csv.head(), '\n')

df['Idade_dobrada'] = df['Idade'].apply(lambda x: x * 2)
print(df, '\n')

media_idade_por_cidade = df.groupby('Cidade')['Idade'].mean()
print(media_idade_por_cidade, '\n')

# Criando uma Series a partir de um dicionário
s2 = pd.Series({'a': 1, 'b': 2, 'c': 3})
print(s2, '\n')

s = pd.Series([1, 3, 5, None, 6, 8], index=['a', 'b', 'c', 'd', 'e', 'f'])

# Indexação direta
print(s['a'], 'Indexação direta')

# Indexação com loc
print(s.loc['b'], 'Indexação com loc \n')

# Slicing com loc
print(s.loc['a':'c'], 'Slicing com loc \n')

# Indexação com iloc
print(s.iloc[2], 'Indexação com iloc \n')

# Slicing com iloc
print(s.iloc[1 : 3], 'Slicing com iloc \n')

print(s.index)   # Exibe os índices
print(s.values)  # Exibe os valores
print(s.dtype)   # Exibe o tipo de dados
