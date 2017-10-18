import numpy as np

class Data:
    def __init__(self, fileName ):
        self.db = np.loadtxt(open(fileName, "rb"), delimiter=",", skiprows=0, dtype=None)
        self.clases = []
        self.baseEntrenamiento =np.array([])
        self.entrenamientoClases = ()
        self.basePrueba = np.array([])
        self.pruebaClases = ()

    def run(self,numCaracteristicas,numClases):

        np.random.shuffle(self.db)

        for i in range(0,numClases):
            self.clases.append([])
        #self.db=np.array(sorted(self.db,key=lambda x:x[numCol-1]))

        for i in range(0,len(self.db)):
            aux=self.db[i][numCaracteristicas]-1
            self.clases[int(aux)].append(self.db[i])
        for j in range(0,len(self.clases)):
            edge = len(self.clases[j])*0.2
            aux=self.clases[j]

            #La matriz aux contiene todas las muestras de la clase j, luego permutamos las filas y tomamos un 20%
            #de estas para el conjunto prueba y el resto para entrenamiento. Realizamos dicha operacion
            #para cada una de las 11 clases
            np.random.shuffle(aux)
            auxPrueba=aux[:int(edge)]
            auxEntrenamiento = aux[int(edge):]
            # TODO: Comentar indices
            #Basicamente anexamos a la matrix las 48 primeras columnas
            self.basePrueba = np.vstack([self.basePrueba,[data[:numCaracteristicas] for data in auxPrueba]]) \
                if len(self.basePrueba) else [data[:numCaracteristicas] for data in auxPrueba]
            self.baseEntrenamiento = np.vstack([self.baseEntrenamiento, [data[:numCaracteristicas] for data in auxEntrenamiento]]) \
                if len(self.baseEntrenamiento) else [data[:numCaracteristicas] for data in auxEntrenamiento]

            #self.pruebaClases = np.hstack([self.pruebaClases,([data[48] for data in auxPrueba])]) if len(self.pruebaClases) else [data[48] for data in auxPrueba]
            self.pruebaClases = np.append(self.pruebaClases,([data[48] for data in auxPrueba]))
            self.entrenamientoClases = np.append(self.entrenamientoClases,([data[48] for data in auxEntrenamiento]))


