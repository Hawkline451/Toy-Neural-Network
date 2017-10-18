import matplotlib.pyplot as plot
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from Data import*

print "Se realizaran los test para un numero n de neuronas en la capa oculta"
figNumber = 1

#Podemos modificar el rango para obtener diferentes resultados, tambien podemos aumentar el numero de
#iteraciones para mejorar el rendimiento (300 iteraciones, son lo suficientemente rapidas y efectivas)

#Probar para range (10,65,10), para revisar rangos mas grandes
for i in range(20,46,5) :
    print "Test " + str(i) + " neuronas"
    neural_network = MLPClassifier(hidden_layer_sizes=(i), solver='adam', max_iter=200, learning_rate='constant')
    d =Data("sensorless_tarea2.txt")

    #Creamos nuestros conjuntos de prueba y entrenamiento para una base de datos con 48 caracteristicas y 11 clases diferentes
    d.run(48,11)
    #'x' contiene las muestras con las 48 caracteristicas 'y' contiene la clase de las muestas
    x_train=d.baseEntrenamiento
    y_train=d.entrenamientoClases
    x_test=d.basePrueba
    y_test=d.pruebaClases

    neural_network.fit(x_train, y_train)

    #Tasa de exito y vector con las clases predichas por la red
    score = neural_network.score(x_test, y_test)
    nnOut = neural_network.predict(x_test)

    #Matriz de confusion para las clases reales (y_test) vs las predichas (nnOut)
    con_matrix=confusion_matrix(y_test, nnOut)

    f= plot.subplot(2,3,figNumber)
    p=plot.imshow(con_matrix, cmap='plasma', interpolation='nearest')
    plot.title(str(i) + " Neuronas, Score:" + str(score))
    plot.colorbar(p)

    figNumber = figNumber + 1

plot.show()
