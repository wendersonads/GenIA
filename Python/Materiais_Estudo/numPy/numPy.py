import numpy as np
import time

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