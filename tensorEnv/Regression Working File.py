import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])
#
# x_train, y_train: Secciones de x e y, respectivamente
# x_test, y_test: Datos para comprobar la precision del modelo creado
# We need to split our data into testing and training data. We will use 90% of our data to train and the other 10% to test.
# The reason we do this is so that we do not test our model on data that it has already seen.
x_train, x_test,y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# Definimos el modelo que vamos a utilizar
linear = linear_model.LinearRegression()

# Entrenamos y ajustamos el resultado
linear.fit(x_train, y_train)
# acc = accuracy
acc = linear.score(x_test, y_test)  # score() Return the coefficient of determination R^2 of the prediction
print(acc)