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


    Methods
    -------
    search():
        Calcula valores aleatorios para "x" y "y", hasta terminar el número de iteraciones y grafica dichos valores
    """

    # Método ejecutado al instanciar la clase, recibe la función objetivo y
    # el rango de valores en el cual se buscará el mínimo global
    def __init__(self, f, xl, xu, yl, yu):
        """
        Inicializa la clase con el rango de valores y la función objetivo

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




    #Función para calcular el mínimo global.
    def search(self):
        """
        Reliza la búsqueda del mínimo global mediante Búsqueda Aleatoria

        Parameters:
            None

        Returns:
            x_best (Numeric): El mejor valor en "x" donde se encuentra el mínimo global
            y_best (Numeric): El mejor valor en "y" donde se encuentra el mínimo global
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
            "\n\nBúsqueda Aleatoria: Mínimo global en m={}, b={}, f(x)={}\n".format(
                round(self.x_best,2), round(self.y_best,2), round(self.f_best[0],2)
            )
        )

        return self.x_best, self.y_best


# Clase para el método de Gradiente Descendente
class DescGradient:
    """
    Clase que ejecuta el método de Gradiente descendente
    para calcular el mínimo global de una función



    Attributes
    ----------
    f: function
        Función objetivo, de la cual se calculará el mínimo global
    xl: Numeric
        Límite inferior en "x"
    xu: Numeric
        Límite superior en "x"
    yl: Numeric
        Límite inferior en "y"
    yu: Numeric
        Límite superior en "y"
    h: Numeric
        Factor de cambio para el gradiente
    x0: Numeric
        Valor inicial en "x" donde es posible encontrar el mínimo global.
        Se obtiene viendo la gráfica de la función objetivo
    y0: Numeric
        Valor inicial en "y" donde es posible encontrar el mínimo global.
        Se obtiene viendo la gráfica de la función objetivo
    xi: Numeric
        Valor en "x" actual donde se interpoló que podría esta el mínimo global.
        Irá cambiando en cada iteración y este será finalmente el valor en "x" del mínimo global
    yi: Numeric
        Valor en "y" actual donde se interpoló que podría esta el mínimo global.
        Irá cambiando en cada iteración y este será finalmente el valor en "y" del mínimo global


    Methods
    -------
    search():
        Realiza el método de Gradiente descendente para calcular el mínimo global de la función objetivo.
        Además va graficando los valores encontrados y aproximaciones
    """




    # Método ejecutado al instanciar la clase, recibe la función objetivo y
    # el rango de valores en el cual se buscará el mínimo global
    def __init__(self, f, xl=-10, xu=10, yl=-10, yu=10):
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
        self.xl = xl
        self.xu = xu
        self.yl = yl
        self.yu = yu

        # El factor de cambio para el gradiente. Se usa un valor pequeño
        self.h = 0.1


    # Esta función realiza lo mismo que search(), exceptuando que en esta no se grafica, solo se obtiene
    # el valor del mínimo global. Por esto se puede colocar muchas más iteraciones
    def search(self, x0, y0, grad_func_x, grad_func_y):
        """
        Se realiza la búsqueda del mínimo global mediante Gradiente Descendente

        Parameters:
            x0 (Numeric): Valor incial en "x" donde es posible que se encuentre el mínimo global
            y0 (Numeric): Valor inicial en "y" donde es posible que se encuntre el mínimo global
            grad_func_x (Function): Función que calcula el gradiente para "xi". Es la derivada parcial
                de la función objetivo con respecto a "x"
            grad_func_y (Function): Función que calcula el gradiente para "yi". Es la derivada parcial
                de la función objetivo con respecto a "y"

        Returns:
            xi (Numeric): Valor en "x" donde se encuentra el mínimo global
            yi (Numeric): Valor en "y" donde se encuentra el mínimo global
        """

        self.x0 = x0
        self.y0 = y0
        self.xi = self.x0
        self.yi = self.y0

        # - Se itera calculando un mejor valor en cada iteración en base al anterior.
        # - Entre más iteraciones se hagan más probable es encontrar un resultado más cercano al mínimo global.
        # - Se coloca un número pequeño de iteraciones porque al graficar y poder ver
        #   el punto más acercado al resultado se necesitan realizar pequeñas pausas.
        for i in range(100000):

            # - Se calcula un (x,y) en base al anterior (xi,yi).
            # - Este servirá para la siguiente iteración
            # - Aquí se manda llamar a la función de cálculo de gradiente
            x_next = self.xi - self.h * grad_func_x(self.xi, self.yi)
            y_next = self.yi - self.h * grad_func_y(self.xi, self.yi)

            # Se actualiza el valor actual como el calculado
            self.xi = x_next
            self.yi = y_next

        print(
            "\n\nGradiente descendente: Mínimo global en x={}, y={}, f(x)={}\n\n".format(
                round(self.xi,2), round(self.yi,2), round(self.f(self.xi, self.yi)[0],2)
            )
        )

        return self.xi, self.yi





# Cargamos los datos para nuestra regresión lineal
data = loadmat('src/exercise_2.mat')

# El tamaño del vector "x" de los datos
n = len(data['X'])

# Separamos los datos en "x" y "y"
X = data['X']
Y = data['Y']


#Función objetivo
def f(m,b):
    value = (1/(2*n)) * sum((Y - (X*m+b))**2)
    
    return value


#--------------------------- Búsqueda Aleatoria ----------------------------------
#random_search = RandomSearch(f, xl=0, xu=2, yl=0, yu=1)
#m,b = random_search.search()


#------------------------------ Gradiente descendiente ----------------------------

# Derivada parcial de la función objetivo con respecto a "m"
def gradient_func_m(m,b):
    
    value = -(1/n) * ((Y - X) * m + b * X).sum()
    return value 


# Derivada parcial de la función objetivo con respecto a "b"
def gradient_func_b(m,b):
    
    value = -(1/n) * (Y-(X * m + b)).sum()
    return value 


gradient = DescGradient(f, xl=0, xu=2, yl=0, yu=1)
m,b = gradient.search(0.8, 0.2, gradient_func_m, gradient_func_b)



# ---------------------------- Graficación -------------------------------

# Se crea un conjunto de valores entre el rango para graficar la linea de
# la regresión lineal
xp = np.arange(-5,15,0.1)

# Se obtiene los valores en "y" en base a "m" y "b" obtenidos con los métodos anteriores 
yp = xp*m+b

# Ploteamos
plt.title('Regresión lineal')
plt.plot(X,Y, 'ro', label="Muestras")
plt.plot(xp,yp, 'b-', label="Regresión")
plt.legend(loc="upper left")
plt.show()