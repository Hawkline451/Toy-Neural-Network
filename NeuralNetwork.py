import matplotlib.pyplot as plot
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from Data import*

#Se escogieron los parametros que se consideraron mas adecuados luego de realizar los test
numNeuronas = 28
neural_network = MLPClassifier(hidden_layer_sizes=(numNeuronas), solver='adam', max_iter=400, learning_rate='constant')

d =Data("sensorless_tarea2.txt")

#Creamos nuestros conjuntos de prueba y entrenamiento para una base de datos con 48 caracteristicas y 11 clases diferentes
d.run(48,11)
x_train=d.baseEntrenamiento
y_train=d.entrenamientoClases
x_test=d.basePrueba
y_test=d.pruebaClases

neural_network.fit(x_train,y_train)

#Imprime en pantalla la tasa de exito al clasificas
print neural_network.score(x_test, y_test)
#Retorna un arreglo con los valores que entrego la red apra un cnjunto de prueba
nnOut = neural_network.predict(x_test)

#Podemos tratar de predecir un cojunto de muestras o solo 1, descomentar para revisar, asd deberia ser de la clase 4
#asd=np.array([1.5797e-06,-2.3624e-07,4.062e-07,4.2115e-06,-2.3139e-06,9.152e-06,0.017096,0.017096,0.017096,0.04659,0.046592,0.046583,0.00080323,0.00027572,0.00075072,0.00082055,0.00028882,0.0004301,1.7761,1.7761,1.776,1.7711,1.7711,1.7711,0.003983,-0.10189,1.55,-0.0082038,-0.22883,0.14454,0.00050089,0.00050364,0.00049311,-0.0020062,-0.0020056,-0.0020108,-0.48154,4.9733,30.742,-0.45664,4.8703,4.7596,-1.4991,-1.4991,-1.4991,-1.4986,-1.4986,-1.4986])
#print neural_network.predict(asd)

#Encontramos la matriz de confusion comparando y_test (clases reales) con nnOut (clases encontradas por la red)
con_matrix=confusion_matrix(y_test, nnOut)
score= neural_network.score(x_test,y_test)

#print con_matrix
f= plot.figure(1)
p=plot.imshow(con_matrix, cmap='plasma', interpolation='nearest')
f.suptitle(str(numNeuronas) + " Neuronas, Score:" + str(score))
plot.colorbar(p)

#Revisamos la tasa de perdida, mientras, mas ajustada a los ejes se encuentre la curva, mejor entrenada estara la red
f2= plot.figure(2)
p2=plot.plot(neural_network.loss_curve_)
plot.xlabel("Steps")
plot.ylabel("Loss Function")


plot.show()


