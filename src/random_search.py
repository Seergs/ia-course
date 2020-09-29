"""

Encuentra el mínimo global de una función

Classes:

    RandomSearch
    DescGradient

Variables

    random_search
    gradient


"""

# Este programa calcula el mínimo global de una función usando los métodos de 
# Búsqueda Aleatoria y Gradiente descendiente
# Para más información sobre el método de Gradiente descendente visite
# https://www.iartificial.net/gradiente-descendiente-para-aprendizaje-automatico/#:~:text=funci%C3%B3n%20de%20coste.-,M%C3%A9todo%20del%20Gradiente%20Descendiente,%2C%20el%20deep%20learning%2C%20etc.



import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import math
from random import random
from mpl_toolkits import mplot3d
import time

# Función objetivo 1
def f1(x, y):
    """
    Evalúa la función objetivo en x,y

        Parameters:
            x (Numeric): coordenada o valor en el eje x
            y (Numeric): coordenada o valor en el eje y
        
        Returns:
            z (Numeric): valor de la función en x,y

    """

    # Evaluamos la función objetivo en x,y
    z = x * (np.e ** (-(x ** 2) - y ** 2))

    return z



# Función que calcula el gradiente usando derivadas parciales
# En este caso es en base a "x". Esta derivada parcial se calculó
# en Symbolab.
def get_gradient_f1_x(x, y):
    """
    Calcula el gradiente usando la derivada parciale en base a "x" de
    la función objetivo

        Parameters: 
            x (Numeric): valor del actual "x", mejor conocido como xi
            y (Numeric): valor del actual "y", mejor conocido como yi

        Returns:
            value (Numeric): el valor del gradiente en x (xi,yi)
    """

    # Calculamos el valor mediante la derivada parcial con respecto a "x"
    value = np.e ** (-(x ** 2) - y ** 2) - 2 * np.e ** (-(x ** 2) - y ** 2) * x ** 2

    return value



# Función que calcula el gradiente usando derivadas parciales
# En este caso es en base a "y". Esta derivada parcial se calculó
# en Symbolab.
def get_gradient_f1_y(x, y):
    """
    Calcula el gradiente usando la derivada parciale en base a "y" de
    la función objetivo

        Parameters: 
            x (Numeric): valor del actual "x", mejor conocido como xi
            y (Numeric): valor del actual "y", mejor conocido como yi

        Returns:
            value (Numeric): el valor del gradiente en x (xi,yi)
    """

    # Calculamos el valor mediante la derivada parcial con respecto a "y"
    value = -2 * np.e ** (-(x ** 2) - y ** 2) * x * y

    return value



# Función objetivo 2
def f2(x1, x2):
    """
    Evalúa la función objetivo en x,y

        Parameters:
            x (Numeric): coordenada o valor en el eje x
            y (Numeric): coordenada o valor en el eje y
        
        Returns:
            z (Numeric): valor de la función en x,y

    """

    # Evaluamos la función objetivo en x,y
    z = (x1 - 2) ** 2 + (x2 - 2) ** 2

    return z


# Función que calcula el gradiente usando derivadas parciales
# En este caso es en base a "x"
def get_gradient_f2_x(x, y):
    """
    Calcula el gradiente usando la derivada parciale en base a "x" de
    la función objetivo

        Parameters: 
            x (Numeric): valor del actual "x", mejor conocido como xi
            y (Numeric): valor del actual "y", mejor conocido como yi

        Returns:
            value (Numeric): el valor del gradiente en x (xi,yi)
    """

    # Calculamos el valor mediante la derivada parcial con respecto a "x"
    value = 2 * (x - 2)

    return value



# Función que calcula el gradiente usando derivadas parciales
# En este caso es en base a "x"
def get_gradient_f2_y(x, y):
    """
    Calcula el gradiente usando la derivada parciale en base a "y" de
    la función objetivo

        Parameters: 
            x (Numeric): valor del actual "x", mejor conocido como xi
            y (Numeric): valor del actual "y", mejor conocido como yi

        Returns:
            value (Numeric): el valor del gradiente en x (xi,yi)
    """

    # Calculamos el valor mediante la derivada parcial con respecto a "y"
    value = 2 * (y - 2)
    return value



# Clase para el método de búsqueda aleatoria
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
    def __init__(self, f, xl=-10, xu=10, yl=-10, yu=10):
        self.f = f
        self.x = 0
        self.y = 0
        self.x_best = 0
        self.y_best = 0
        self.f_best = np.inf
        self.xl = xl
        self.xu = xu
        self.yl = yl
        self.yu = yu

        self.x_range = np.arange(xl, xu, 0.1)
        self.y_range = np.arange(xl, xu, 0.1)
        self.X, self.Y = np.meshgrid(self.x_range, self.y_range)
        self.Z = self.f(self.X, self.Y)

    def search(self):
        for i in range(100):
            self.x = self.xl + (self.xu - self.xl) * random()
            self.y = self.yl + (self.yu - self.yl) * random()

            f_val = self.f(self.x, self.y)
            if f_val < self.f_best:
                self.f_best = f_val
                self.x_best = self.x
                self.y_best = self.y

            self.__plot_contour()

            print("Ploteando {}, {}".format(self.x, self.y))

        print(
            "Mínimo global en x={}, y={}, f(x)={}".format(
                round(self.x_best, 2), round(self.y_best, 2), round(self.f_best, 2)
            )
        )
        plt.show()

    def search_lite(self):
        for i in range(1000000):
            self.x = self.xl + (self.xu - self.xl) * random()
            self.y = self.yl + (self.yu - self.yl) * random()

            f_val = self.f(self.x, self.y)
            if f_val < self.f_best:
                self.f_best = f_val
                self.x_best = self.x
                self.y_best = self.y

        print(
            "Mínimo global en x={}, y={}, f(x)={}".format(
                round(self.x_best, 2), round(self.y_best, 2), round(self.f_best, 2)
            )
        )

    def plot_surface(self):
        plt.figure()
        ax = plt.axes(projection="3d")
        ax.plot_surface(self.X, self.Y, self.Z)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()

    def __plot_contour(self):
        plt.clf()
        plt.contour(self.X, self.Y, self.Z)

        self.__plot_point(self.x, self.y, "bo")
        self.__plot_point(self.x_best, self.y_best, "go")
        plt.pause(0.005)


    def __plot_point(self, x, y, marker):
        if marker == "go":
            label = "{:.2f}".format(y)
            plt.annotate(
                label, (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
            )
        plt.plot(x, y, marker)


class DescGradient:
    def __init__(self, f, xl=-10, xu=10, yl=-10, yu=10):
        self.f = f
        self.xl = xl
        self.xu = xu
        self.yl = yl
        self.yu = yu

        self.h = 0.1

        self.x_range = np.arange(xl, xu, 0.1)
        self.y_range = np.arange(xl, xu, 0.1)
        self.X, self.Y = np.meshgrid(self.x_range, self.y_range)
        self.Z = self.f(self.X, self.Y)

    def plot_surface(self):
        plt.figure()
        ax = plt.axes(projection="3d")
        ax.plot_surface(self.X, self.Y, self.Z)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()

    def __plot_contour(self):
        plt.clf()
        plt.contour(self.X, self.Y, self.Z)

        self.__plot_point(self.x0, self.y0, "bo")
        self.__plot_point(self.xi, self.yi, "go")
        plt.pause(0.005)

    def __plot_point(self, x, y, marker):
        if marker == "go":
            label = "{:.2f}".format(y)
            plt.annotate(
                label, (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
            )
        plt.plot(x, y, marker)

    def search(self, x0, y0, grad_func_x, grad_func_y):
        self.x0 = x0
        self.y0 = y0
        self.xi = self.x0
        self.yi = self.y0
        for i in range(100):
            self.x_next = self.xi - self.h * grad_func_x(self.xi, self.yi)
            self.y_next = self.yi - self.h * grad_func_y(self.xi, self.yi)
            self.xi = self.x_next
            self.yi = self.y_next

            print("Ploteando {},{}".format(self.xi, self.yi))

            self.__plot_contour()

        print(
            "Mínimo global en x={}, y={}, f(x)={}".format(
                round(self.xi, 2), round(self.yi, 2), round(self.f(self.xi, self.yi), 2)
            )
        )

    def search_lite(self, x0, y0, grad_func_x, grad_func_y):
        self.x0 = x0
        self.y0 = y0
        self.xi = self.x0
        self.yi = self.y0
        for i in range(10000):
            self.x_next = self.xi - self.h * grad_func_x(self.xi, self.yi)
            self.y_next = self.yi - self.h * grad_func_y(self.xi, self.yi)
            self.xi = self.x_next
            self.yi = self.y_next

        print(
            "Mínimo global en x={}, y={}, f(x)={}".format(
                round(self.xi, 2), round(self.yi, 2), round(self.f(self.xi, self.yi), 2)
            )
        )


# --------Busqueda aleatoria-------

# Primera funcion
# random_search = RandomSearch(f1, xl=-2, xu=2, yl=-2, yu=2)
# random_search.search_lite()
# random_search.plot_surface()

# Segunda funcion
# random_search = RandomSearch(f2, xl=1, xu=2, yl=1, yu=2)
# random_search.search_lite()
# random_search.plot_surface()


# -------Gradiente---------


# Primera funcion
# gradient = DescGradient(f1, xl=-2, xu=2, yl=-2, yu=2)
# gradient.search(-0.5,0, get_gradient_f1_x, get_gradient_f1_y)
# gradient.plot_surface()


# Segunda funcion
# gradient = DescGradient(f2, xl=1, xu=2, yl=1, yu=2 )
# gradient.search(1.8,1.8,get_gradient_f2_x, get_gradient_f2_y)
# gradient.plot_surface()
