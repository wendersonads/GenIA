import numpy as np
import time
import matplotlib.pyplot as plt

print("Num Py Version", np.__version__)

a = np.array([1,2,3,4,5])

b = a * 2
print("Array a:", a, "Array b:", b)

for pos, arr in enumerate(a):
    if arr == 2:
        print("Posição no Array", pos)

c = np.array([[1,2], [3,4]])
print("Matriz c:\n", c)

d = c + 3
print("Matriz d:\n", d)

media = np.mean(a)
print("Média dos elementos de a: ", media)

zeros_array = np.zeros((3,3))
print("Array zeros_array:\n", zeros_array)

range_array = np.arange(0, 10, 2)
print("Array com np.arange:", range_array)

linspace_array = np.linspace(0, 10, 5)
print("Array com np.linspace:", linspace_array)
print("Forma do array:", linspace_array.shape)
print("Número de dimensões:", linspace_array.ndim)

matriz = np.array([[1, 2, 3], [4, 5, 6]])
print("Matriz:\n", matriz)

print("Tipo de dados de 'a':", a.dtype)

# Lista: Mutável, pode conter elementos de diferentes tipos
lista = [1, 'dois', 3.0]
lista.append(4)  # Adiciona um elemento ao final
print("Lista modificada:", lista)
# Saída esperada: Lista modificada: [1, 'dois', 3.0, 4]

# Tupla: Imutável, pode conter elementos de diferentes tipos
tupla = (1, 'dois', 3.0)
# tupla[0] = 2  # Isto resultaria em um erro
print("Tupla:", tupla)
# Saída esperada: Tupla: (1, 'dois', 3.0)

# Conjunto: Não ordenado, sem duplicatas
conjunto = {1, 2, 2, 3, 4, 5, 5, 5, 5, 6} # ignora os outros '5' e imprimi somente 5
print("Conjunto (sem duplicatas):", conjunto)
# Saída esperada: Conjunto (sem duplicatas): {1, 2, 3, 4, 5, 6}

lista = list(range(1000000))
inicio = time.time()
lista += [x + 1 for x in lista]
fim = time.time()
print("Tempo com lista:", fim - inicio)

array = np.arange(1000000)
inicio = time.time()
array += 1
fim = time.time()
print("Tempo com array NumPy:", fim - inicio)
# Saida Esperada (os resultados podem variar pois dependem das configurações de velocidade de processador e memória):
# Tempo com lista: 0.23276233673095703
# Tempo com array NumPy: 0.002035856246948242

a = np.array([1,2,3])
b = np.array([4,5,6])

#Adição
c = a + b
print("Adição:", c)

d = b - a
print("Subtração:", d)

e = a * b
print("Multiplicação:", e)

f = b / a
print("Divisao:", f)

h = np.array([1, 2, 3])
i = np.array([[0], [1], [2]])
# Broadcasting entre diferentes tamanhos
j = h + i
print("Broadcasting entre diferentes tamanhos:\n", j)
# Saída esperada:
# Broadcasting entre diferentes tamanhos:
# [[1 2 3]
#  [2 3 4]
#  [3 4 5]]

# Indexação simples
k = a[1]
print("Elemento no índice 1 -", k)
# Saída esperada: Elemento no índice 1 - 2

# Slicing
l = a[0 : 2]
print("Primeiros dois elementos:", l)
# Saída esperada: Primeiros dois elementos: [1 2]

m = np.array([1, 2, 3, 4, 5])
m[1 : 4] = 0
print("Array após zeroing:", m)
# Saída esperada: Array após zeroing: [1 0 0 0 5]


# Slicing com step
m = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
step_slice = m[ 1 : 8 : 2]
print("Elementos selecionados com step de 2 -", step_slice)
# Saída esperada: Elementos selecionados com step de 2 - [1 3 5 7]

# Usando step negativo para inverter o array
reverse_slice = m[::-1]
print("Array invertido:", reverse_slice)
# Saída esperada: Array invertido: [9 8 7 6 5 4 3 2 1 0]

n = np.array([10, 20, 30, 40, 50])
indices = [1, 3, 4]
o = n[indices]
print("Elementos selecionados:", o)
# Saída esperada: Elementos selecionados: [20 40 50]

p = np.array([1, 2, 3, 4, 5])
mask = p > 3
q = p[mask]
print("Elementos maiores que 3 -", q)
# Saída esperada: Elementos maiores que 3 - [4 5]

r = np.array([1, 2, 3, 4, 5])
s = np.where(r < 3, 0, r)
print("Substituição condicional:", s)
# Saída esperada: Substituição condicional: [0 0 3 4 5]


A = np.array([
    [1, 2],
    [3, 4]
    ])
B = np.array([
    [2, 0],
    [1, 2]
])
# Produto de matrizes
produto = np.dot(A, B)
print("Produto das matrizes:\n", produto)
# Saída esperada:
# Produto das matrizes:
# [[ 4  4]
#  [10  6]]

# Determinante
determinante = np.linalg.det(A)
print("Determinante:", determinante)
# Saída esperada: Determinante: -2.0

# Autovalores
autovalores = np.linalg.eigvals(A)
print("Autovalores:", autovalores)
# Saída esperada: Autovalores: [-0.37228132  5.37228132]

dados = np.array([1, 2, 3, 4, 5])
# Média
media = np.mean(dados)
print("Média:", media)
# Saída esperada: Média: 3.0

# Mediana
mediana = np.median(dados)
print("Mediana:", mediana)
# Saída esperada: Mediana: 3.0

# Desvio padrão
desvio_padrao = np.std(dados)
print("Desvio padrão:", desvio_padrao)
# Saída esperada: Desvio padrão: 1.4142135623730951

# Dados
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Plotando
plt.figure(figsize=(8, 4))
plt.plot(x, y, label='sin(x)')
plt.title('Gráfico de Linha Simples')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True)
plt.show()

# Dados aleatórios
data = np.random.normal(size=1000)
# Histograma
plt.figure(figsize=(8, 4))
plt.hist(data, bins=30, alpha=0.75, color='blue', edgecolor='black')
plt.title('Histograma de Dados Aleatórios')
plt.xlabel('Valores')
plt.ylabel('Frequência')
plt.grid(True)
plt.show()