from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Carrega o conjunto de dados Iris
iris = load_iris()
X, y = iris.data, iris.target

# Divide os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Cria e treina o modelo RDA (usando LDA com shrinkage)
rda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
rda.fit(X_train, y_train)

# Realiza previsões e avalia a acurácia
accuracy = rda.score(X_test, y_test)
print(f"Acurácia do RDA: {accuracy:.2f}")