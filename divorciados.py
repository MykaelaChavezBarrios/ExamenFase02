# AUTHOR: TATYANA MYKAELA CHAVEZ BARRIOS
# tratamiento de datos
import pandas as pd
import numpy as np

# entrenamiento
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# importando la data
data = pd.read_csv("content/divorcios.csv", header=0, sep=";")
print(data.head(10))
print(data.shape)
# porcentaje de personas divorciados
print(np.average(data.Divorcio) * 100)

# asignando valores x y
x = data.iloc[:, [0, 1, 2, 3, 4]].values
y = data.iloc[:, 5].values
# separacion train y test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
# escalado de variables
standard_x = StandardScaler()
x_train = standard_x.fit_transform(x_train)
x_test = standard_x.fit_transform(x_test)
# entrenando el modelo
reg = LogisticRegression()
reg = reg.fit(x_train, y_train)
# prediccion
y_pred = reg.predict(x_test)
print("Accuracy de entrenamiento:", reg.score(x_train, y_train))
