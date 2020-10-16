from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from random import random
from mpl_toolkits import mplot3d
import time

class RandomSearch:
    """
    Clase que ejecuta el método de búsqueda aleatoria
    para calcular le mínimo global de una función


    Attributes
    ----------
    f: function
        Función objetivo, de la cual se calculará el mínimo global
    x: Numeric
        Valor en "x" de la búsqueda aleatoria, irá cambiando en cada iteración
    y: Numeric
        Valor en "y" de la búsqueda aleatoria, irá cambiando en cada iteración
    x_best: Numeric
        Mejor valor en "x", cada vez que se encuentre un mejor valor (más cerca del mínimo global) se actualizará
    y_best: Numeric
        Mejor valor en "y", cada vez que se encuentre un mejor valor (más cerca del mínimo global) se actualizará
    f_best: Numeric
        Mejor valor en "z" o valor evaluado en la función objetivo,
        cada vez que se encuentre un mejor valor (más cerca del mínimo global) se actualizará
    xl: Numeric
        Límite inferior en "x"
    xu: Numeric
        Límite superior en "x"
    yl: Numeric
        Límite inferior en "y"
    yu: Numeric
        Límite superior en "y"
    x_range: ndarray (numpy)
        Rango de valores entre el límite inferior y superior en "x", usado para graficar
    y_range: ndarray (numpy)
        Rango de valores entre el límite inferior y superior en "y", usado para graficar
    X: array (numpy)
        Lista de una dimensión que representa las coordenadas de una matriz, para el vector "x"
    Y: array (numpy)
        Lista de una dimensión que representa las coordenadas de una matriz, para el vector "y"
    Z: array
        Lista de los valores de la función objetivos evaluada en X,Y


    Methods
    -------
    search():
        Calcula valores aleatorios para "x" y "y", hasta terminar el número de iteraciones y grafica dichos valores
    search_lite():
        Calcula valores aleatorios para "x" y "y", hasta terminar el número de iteraciones
    plot_surface():
        Grafica en 3D la fución objetivo en el rango especificado
    __plot_contour():
        Grafica la función objetivo en el rango especificado
    __plot_point(x,y,marker):
        Grafica un punto en el contour
    """

    # Método ejecutado al instanciar la clase, recibe la función objetivo y
    # el rango de valores en el cual se buscará el mínimo global
    def __init__(self, f, xl, xu, yl, yu):
        """
        Inicializa la clase con el rango de valores, la función objetivo
        y los vectores a graficar

        Parameters:
            f (function): Función que retorna el valor al evaluar la función objetivo en un punto
            xl (Numeric): Límite inferior en "x"
            xu (Numeric): Límite superior en "x"
            yl (Numeric): Límite inferior en "y"
            yu (Numeric): Límite superior en "y"

        Returns:
            None
        """

        self.f = f

        # Inicializamos el vector (x,y) en (0,0)
        self.x = 0
        self.y = 0

        # Igualmente igualamos el vector (x_best, y_best) en (0,0)
        # Nota: Este será finalmente la coordenada en (x,y) del
        # mínimo global de la función y al evaluarla en estos puntos
        # obtendremos nuestra f_best
        self.x_best = 0
        self.y_best = 0

        # Será finalmente el valor de la función objetivo evaluada
        # en el punto (x,y) = (x_best, y_best).
        # Se incializa en infinito (np.inf) para que cualquier valor
        # que sea el incial se menor que este, así se irá ajustando
        self.f_best = np.inf

        self.xl = xl
        self.xu = xu
        self.yl = yl
        self.yu = yu




    # Esta función realiza lo mismo que search(), exceptuando que en esta no se grafica, solo se obtiene
    # el valor del mínimo global. Por esto se puede colocar muchas más iteraciones
    def search(self):
        """
        Reliza la búsqueda del mínimo global mediante Búsqueda Aleatoria

        Parameters:
            None

        Returns:
            None
        """

        # - Se itera calculando un valor aleatorio.
        # - Entre más iteraciones se hagan más probable es encontrar un resultado más cercano al mínimo global.
        # - Este método utiliza fuerza bruta.
        for i in range(100000):

            # Se calcula una posición (x,y) aleatoria
            self.x = self.xl + (self.xu - self.xl) * random()
            self.y = self.yl + (self.yu - self.yl) * random()

            # Se evalua la función objetivo en el punto (x,y)
            f_val = self.f(self.x, self.y)

            # Si el valor que se obtuvo es menor al que ya se consideraba "menor" (f_best),
            # entonces este valor se convierte en el nuevo punto más acertado en (x,y,f_best).
            if f_val < self.f_best:
                self.f_best = f_val
                self.x_best = self.x
                self.y_best = self.y

        print(
            "\n\nMínimo global en m={}, b={}, f(x)={}\n".format(
                self.x_best, self.y_best, self.f_best
            )
        )

        return self.x_best, self.y_best


data = loadmat('src/exercise_2.mat')
n = len(data['X'])

X = data['X']
Y = data['Y']


def f(m,b):
    z = (1/(2*n)) * sum((Y - (X*m+b))**2)
    
    return z

random_search = RandomSearch(f, xl=0, xu=2, yl=0, yu=1)

m,b = random_search.search()
xp = np.arange(-5,15,0.1)
yp = xp*m+b

plt.title('Regresión lineal')
plt.plot(X,Y, 'ro', label="Muestras")
plt.plot(xp,yp, 'b-', label="Regresión")
plt.legend(loc="upper left")
plt.show()


