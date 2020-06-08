
# En el siguiente programa se trata de predecir el resultado final (G3) de un conjunto de estudiantes
# portugueses en torno a dos asignaturas: matematicas y portugués. Se realiza un estudio trabajando con 5 dimensiones:
# G1 : Nota del primer trimestre. numeric: from 0 to 20
# G2 : Nota del segundo trimestre. numeric: from 0 to 20
# studytime : Horas de estudio. numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours
# failures : Veces que se han suspendido las asignaturas. numeric: n if 1<=n<3, else 4
# absences : Veces que no se ha asistido a la escuela. numeric: from 0 to 93

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# Leemos el dataset del fichero csv
data = pd.read_csv("student-mat.csv", sep=";")

# Seleccionamos las dimensiones
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# Dimension a predecir
predict = "G3"

# Quitamos la columna G3
X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# x_train, y_train: Secciones de x e y, respectivamente
# x_test, y_test: Datos para comprobar la precision del modelo creado
# Separamos los datos entre datos de entrenamiento y datos de prueba. Usaremos 90% de nuestros datos como entrenamiento
# y el 10% restante de prueba.
# Hacemos esto para que no probemos el modelo con datos que ya se han visto.
x_train, x_test,y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

'''
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    # Definimos el modelo que vamos a utilizar
    linear = linear_model.LinearRegression()

    # Entrenamos y ajustamos el resultado
    linear.fit(x_train, y_train)
    # acc == accuracy == precision
    acc = linear.score(x_test, y_test)  # score() Return the coefficient of determination R^2 of the prediction
    print('Precission: \n', str(acc))

    if acc > best:
        best = acc
        # Crea y guarda un archivo pickle con el contenido del modelo
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)'''

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print('Coefficient: \n', linear.coef_) # Cada m en y=mx+b | En este caso tenemos 5 dimensiones
print('Intercept: \n', linear.intercept_) # Variable b

# ============== PREDICCIONES ==============

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

# RESULTADO OBTENIDO PARA UN ESTUDIANTE AL AZAR
# EJEMPLO: 12.659181500325518 [13 12  1  0 20] 12
# 12.659181500325518  PREDICCION DE RESULTADO FINAL
# 13                  GRADE 1
# 12                  GRADE 2
# 1                   studytime
# 0                   failures
# 20                  absences
# 12                  NOTA FINAL

# Conseguimos lograr una precision de en torno al 86%


# ============== PLOTTING ==============

p = 'Failures'
style.use("ggplot")
pyplot.scatter(data[p],data['G3'])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()