import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

# Quitamos la columna G3
X = np.array(data.drop([predict], 1))
y = np.array(data[predict])
#
# x_train, y_train: Secciones de x e y, respectivamente
# x_test, y_test: Datos para comprobar la precision del modelo creado
# Separamos los datos entre datos de entrenamiento y datos de prueba. Usaremos 90% de nuestros datos como entrenamiento
# y el 10% restante de prueba.
# Hacemos esto para que probemos el modelo en datos que ya se han visto.
x_train, x_test,y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# Definimos el modelo que vamos a utilizar
linear = linear_model.LinearRegression()

# Entrenamos y ajustamos el resultado
linear.fit(x_train, y_train)
# acc = accuracy
acc = linear.score(x_test, y_test)  # score() Return the coefficient of determination R^2 of the prediction
print(acc)

print('Coefficient: \n', linear.coef_) # These are each slope value // Cada m en y=mx+b
print('Intercept: \n', linear.intercept_) # This is the intercept // La b

# ============== PARA PREDECIR EL RESULTADO DE UN ESTUDIANTE EN CONCRETO ==============

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

