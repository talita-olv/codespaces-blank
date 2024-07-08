# Importando as bibliotecas necessárias
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

# Carregando o conjunto de dados MNIST (dígitos escritos à mão)
digits = datasets.load_digits()

# Dividindo os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# Criando um classificador SVM
classifier = svm.SVC(kernel='linear')

# Treinando o modelo SVM
classifier.fit(X_train, y_train)

# Fazendo previsões nos dados de teste
y_pred = classifier.predict(X_test)

# Avaliando a precisão do modelo
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo SVM: {accuracy:.2f}')
