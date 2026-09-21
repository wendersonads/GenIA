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

# Filtrando elementos maiores que 3
print(s[s > s.mean()])
# Saída esperada:
# 2    5.0
# 4    6.0
# 5    8.0
# dtype: float64

print(s.where(s > 3, 'below threshold'))
# Saída esperada:
# a    below threshold
# b    below threshold
# c                5.0
# d                NaN
# e                6.0
# f                8.0
# dtype: object

print(s.sort_values())  # Ordena os valores
print(s.drop_duplicates())  # Remove duplicatas
print(s.isnull())  # Detecta valores nulos


print(s.reindex(['g', 'f', 'e', 'd', 'c', 'b', 'a']))
# Saída esperada:
# g    NaN
# f    8.0
# e    6.0
# d    NaN
# c    5.0
# b    3.0
# a    1.0
# dtype: float64


data = {'ID': [1, 2, 3],
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35]}
df = pd.DataFrame(data)
print(df, '\n')
# Saída esperada:
#    ID     Name  Age
# 0   1    Alice   25
# 1   2      Bob   30
# 2   3  Charlie   35

# Acessando uma coluna
print(df['Name'], '\n')
# Saída esperada:
# 0      Alice
# 1        Bob
# 2    Charlie
# Name: Name, dtype: object

# Acessando um elemento específico
print(df.loc[0, 'Name'], '\n')
# Saída esperada:
# Alice


# Adicionando uma nova coluna
df['Department'] = ['HR', 'Tech', 'Marketing']
print(df, '\n')
# Saída esperada:
#    ID     Name  Age Department
# 0   1    Alice   25         HR
# 1   2      Bob   30       Tech
# 2   3  Charlie   35  Marketing

# Excluindo uma coluna
# df.drop(
#     'Department',
#     axis=1, # O parâmetro axis determina se a operação afeta linhas (axis=0) ou colunas (axis=1)
#     inplace=True # O parâmetro inplace, quando definido como “True”, aplica as mudanças diretamente no DataFrame original, modificando-o sem criar uma nova cópia. Se inplace=False (o padrão), a operação retorna um novo DataFrame com a modificação, deixando o original inalterado.
# )
print(df, '\n')
# Saída esperada:
#    ID     Name  Age
# 0   1    Alice   25
# 1   2      Bob   30
# 2   3  Charlie   35


# Adicionando uma nova linha
new_row = {'ID': 4, 'Name': 'Dave', 'Age': 28}
df = pd.concat(
    [df, pd.DataFrame([new_row])],
    ignore_index=True
)
print(df, '\n')
# Saída esperada:
#    ID     Name  Age
# 0   1    Alice   25
# 1   2      Bob   30
# 2   3  Charlie   35
# 3   4     Dave   28

# Excluindo uma linha
df.drop(3, inplace=True)
print(df, '\n')
# Saída esperada:
#    ID     Name  Age
# 0   1    Alice   25
# 1   2      Bob   30
# 2   3  Charlie   35

print('Mostra as duas primeiras linhas \n', df.head(2))
print('Mostra as duas últimas linhas \n', df.tail(2))  # Mostra as duas últimas linhas

print('Resumo estatístico da coluna Age \n', df['Age'].describe())  # Resumo estatístico da coluna 'Age'
print('Contagem de valores únicos na coluna Name \n', df['Name'].value_counts())  # Contagem de valores únicos na coluna 'Name'

# Filtrando dados usando query
# Com expressão booleana: df[(df['Age'] > 30) & (df['Name'] == 'Charlie')]
result = df.query("Age > 30 & Name == 'Charlie'")
print('Result Query', result)
# Saída esperada:
#    ID     Name  Age
# 2   3  Charlie   35


# Alterando o índice para a coluna 'Name'
df.set_index('Name', inplace=True)
print('Alterando o índice para a coluna Name \n', df, '\n')
# Saída esperada:
#          ID  Age
# Name
# Alice     1   25
# Bob       2   30
# Charlie   3   35
# Dave      4   28


# Resetando para o índice padrão
df.reset_index(inplace=True)
print('Resetando para o índice padrão \n', df, '\n')
# Saída esperada:
#       Name  ID  Age
# 0    Alice   1   25
# 1      Bob   2   30
# 2  Charlie   3   35
# 3     Dave   4   28

grouped = df.groupby('Department')['Age'].mean() #Os dados do DataFrame df são agrupados pela coluna “Department” sendo feita a média das idades (coluna “Age”) para cada departamento.
print('', grouped, '\n')

data1 = {'ID': [1, 2, 3],
        'Name': ['Lucas', 'Bob', 'Charlie'],
        'Age': [40, 50, 60]}

data2 = {'ID': [1, 2, 3],
         'Name': ['Alice', 'Pabla', 'Charliely'],
         'Age': [18, 20, 25]}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# MERGE
merged_df = df1.merge(
    df2,
    on='ID',
    how='inner'
)
print(merged_df, '\n')

# JOIN
joined_df = df1.join(
    df2,
    lsuffix='_df1',
    rsuffix='_df2',
    how='outer'
)
print(joined_df, '\n')

# Criando os dados
data = {
    'Product': [
        'Notebook', 'Notebook', 'Notebook',
        'Celular', 'Celular', 'Celular',
        'Tablet', 'Tablet', 'Tablet'
    ],
    'Region': [
        'Norte', 'Sul', 'Sudeste',
        'Norte', 'Sul', 'Sudeste',
        'Norte', 'Sul', 'Sudeste'
    ],
    'Sales': [
        1000, 1500, 2000,
        1200, 1800, 2500,
        800, 1100, 1600
    ]
}

pf = pd.DataFrame(data)
print(pf, '\n')

# Criando a tabela dinâmica
pivot = pd.pivot_table(
    pf,
    values='Sales',
    index='Product',
    columns='Region',
    aggfunc='sum'
)

print(pivot, '\n')

df['Age'].fillna(df['Age'].mean())

# Concatenando DataFrames verticalmente (adição de linhas)
new_data_rows = pd.DataFrame({'ID': [6, 7], 'Name': ['Lucas', 'Daniela'], 'Age': [31, 29]})
concatenated_df = pd.concat([df, new_data_rows], ignore_index=True)
print(concatenated_df, '\n')

# Concatenando DataFrames horizontalmente (adição de colunas)
new_data_columns = pd.DataFrame({'Department': ['HR', 'Tech', 'Marketing']})
concatenated_df = pd.concat([df, new_data_columns], axis=1)
print(concatenated_df, '\n')

# Escrevendo para CSV
df.to_csv('output.csv', index=False)

# Lendo de CSV
df_from_csv = pd.read_csv('output.csv')
print(df_from_csv, '\n')


# Suponha que df seja um DataFrame Pandas complexo
df = pd.DataFrame({
    'A': range(1, 6),
    'B': ['A', 'B', 'C', 'D', 'E'],
    'C': pd.date_range('20230101', periods=5)
})

# Serializando o DataFrame usando Pickle
df.to_pickle('dataframe.pkl')

# Desserializando o DataFrame
loaded_df = pd.read_pickle('dataframe.pkl')
print(loaded_df)
# Saída esperada:
#    A  B          C
# 0  1  A 2023-01-01
# 1  2  B 2023-01-02
# 2  3  C 2023-01-03
# 3  4  D 2023-01-04
# 4  5  E 2023-01-05