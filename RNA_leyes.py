import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

carpeta = r"C:/Users/..../" 
archivo = "........xlsx"

df = pd.read_excel(carpeta + archivo)

x = np.array([df["Este [x]"]])
y = np.array([df["Norte [y]"]])
cu = np.array([df["......"]])

marker_size = 15
plt.scatter(x,y , marker_size, cu, cmap=plt.cm.Blues)
plt.xlabel("Este [x]")
plt.ylabel("Norte [y]")
plt.title(".......")

cbar = plt.colorbar()
cbar.set_label(".......", labelpad=+5)
plt.show()

file='x.txt'
x_train=np.loadtxt(file,delimiter='\t', skiprows=0,usecols=[0,1])
#print(x_train)
#print(x_train.shape)

file2='y.txt'
y_train=np.loadtxt(file2,delimiter='\t', skiprows=0,usecols=[0])
#print(y_train)
#print(y_train.shape)

# Usar un mínimo del 20% en test
file3='x - test.txt'
x_test=np.loadtxt(file3,delimiter='\t', skiprows=0,usecols=[0,1])

file4='y - test.txt'
y_test=np.loadtxt(file4,delimiter='\t', skiprows=0,usecols=[0])

##################################################################################
############################# RED NEURONAL #######################################
##################################################################################

from keras.models import Sequential
from keras.layers import Dense, Activation

#reservar filas para validar los datos
x_val = x_train[70:,]
y_val = y_train[70:,]

#print(x_train[:3,:])

#1-Capas de entrada
#2-Capas ocultas
#3-Capas de Salida

model = Sequential()
#Dos neuronas de entrada
model.add(Dense(2, input_dim = 2, kernel_initializer='normal', activation='softsign')) #REALIZAMOS NUESTRA NEURONA
#Seis neuronas de capas ocultas
model.add(Dense(6, kernel_initializer='normal', activation='softsign'))
#una neurona de capa de salida
model.add(Dense(1, kernel_initializer='normal')) # NEURONA DE SALIDA CON 1 NEURONA

# Buscar mas funciones en: Layer activation functions
#https://keras.io/api/layers/activations/

####################################### Entrenamiento #######################################

#Procedimiento
#1-Compilar Modelo
#2-Entrenar el modelo
#3-Evaluar Modelo
#4-Testear Modelo

#######################################         1-Compilar Modelo

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_percentage_error']) #

#SE PUEDE CAMBIAR METRICA A metrics= accuracy, binary_crossentropy o mean_absolute_percentage_error
#https://keras.io/api/metrics/probabilistic_metrics/#binarycrossentropy-class


#######################################         2-Entrenar el modelo

model.fit(x_train, y_train, batch_size=10, epochs=50, validation_data=(x_val, y_val))

#######################################         3-Evaluar Modelo

resultados=model.evaluate(x_test, y_test)

# LOSS=ERROR EN LA SALIDA
# MEAN_ABSOLUTE% =ERROR ABSOLUTO

for i in range(len(model.metrics_names)):
    print(model.metrics_names[i],":",resultados[i])

# aqui generamos nuestra prediccion con los del test
prediccion=model.predict(........) #Utilizar conjunto de datos a convenir
print(prediccion)

prediccion[10:]