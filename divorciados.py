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
