# 
# Sergio Suárez Álvarez
# 217758497
# Seminario de solución de inteligencia Artificial I
#
#
# Este programa calcula el mínimo global de una función usando los métodos de 
# Búsqueda Aleatoria y Gradiente descendiente
# Para más información sobre el método de Gradiente descendente visite
# https://www.iartificial.net/gradiente-descendiente-para-aprendizaje-automatico/#:~:text=funci%C3%B3n%20de%20coste.-,M%C3%A9todo%20del%20Gradiente%20Descendiente,%2C%20el%20deep%20learning%2C%20etc.
#
#
#


"""

Encuentra el mínimo global de una función

Classes:

    RandomSearch
    DescGradient

Variables

    random_search
    gradient


"""


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

        # Rango de valores entre el límite inferior y superior con una separación de 0.1
        # Se crea un rango tanto para "x" como para "y"
        self.x_range = np.arange(xl, xu, 0.1)
        self.y_range = np.arange(xl, xu, 0.1)

        # Nuestros vectores "x" y "y"
        self.X, self.Y = np.meshgrid(self.x_range, self.y_range)

        # Calculamos nuestro vector "z" evaluando en la función objetivo "x" y "y"
        self.Z = self.f(self.X, self.Y)


    # Función que realiza la Búsqueda Aleatoria para encontrar el mínimo global en
    # la función objetivo, además de ir graficando tanto el punto aleatorio como el
    # mejor punto encontrado hasta el momento
    def search(self):
        """
        Se realiza la búsqueda del mínimo global mediante Búsqueda Aleatoria y se
        van graficando los resultados 

        Parameters:
            None
        
        Returns:
            None
        """


        # - Se itera calculando un valor aleatorio. 
        # - Entre más iteraciones se hagan más probable es encontrar un resultado más cercano al mínimo global.
        # - Este método utiliza fuerza bruta.
        # - Se coloca un número pequeño de iteraciones porque al graficar y poder ver 
        #   tanto el punto más acercado al resultado y el punto aleatoria se necesitan realizar pequeñas pausas.
        for i in range(100):

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

            # Graficamos el contour de la función. Dentro de esta se grafica también el valor
            # aleatorio obtenido y el mejor valor hasta el momento
            self.__plot_contour()

            print("Ploteando {}, {}".format(self.x, self.y))

        print(
            "Mínimo global en x={}, y={}, f(x)={}".format(
                round(self.x_best, 2), round(self.y_best, 2), round(self.f_best, 2)
            )
        )


        # Esta línea es importante porque si no se pone, al terminar las iteraciones la gráfica
        # de contour con el punto f_best o mínimo global se cerrará
        plt.show()



    # Esta función realiza lo mismo que search(), exceptuando que en esta no se grafica, solo se obtiene
    # el valor del mínimo global. Por esto se puede colocar muchas más iteraciones
    def search_lite(self):
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
        for i in range(1000000):

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
            "Mínimo global en x={}, y={}, f(x)={}".format(
                round(self.x_best, 2), round(self.y_best, 2), round(self.f_best, 2)
            )
        )



    # Función para graficar la superficie en 3D de la función objetivo en el rango especificado
    def plot_surface(self):
        """
        Grafica la superficie en 3D de la función ebjetivo en el rango especificado

        Parameters:
            None

        Returns:
            None
        """

        # Se crea una ventana nueva para graficar.
        # Esto porque el contour puede ya estar en pantalla
        plt.figure()

        # Le decimos a la libreria de graficas que queremos la proyección en 3D
        ax = plt.axes(projection="3d")

        # Graficamos
        ax.plot_surface(self.X, self.Y, self.Z)

        # Se ponen títulos a los ejes
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        # Mostramos la gráfica
        plt.show()




    # - Función que grafica el contour de la función objetivo en el rango especificado,
    #   el punto aleatorio obtenido en search() y el mejor punto encontrado hasta el momento (x_best, y_best) 
    # - Esta función será llamada iterativamente
    def __plot_contour(self):
        """
        Grafica la funcion objetivo en el rango especificado y los puntos (x,y) y (x_best, y_best)
        -------
        Private
        -------

        Parameters:
            None
        
        Returns:    
            None
        """

        # Limpiamos el previo marcador que se había ploteado en la iteración anterior
        plt.clf()

        # Graficamos nuestro contour con los vectores "x", "y" y "z"
        plt.contour(self.X, self.Y, self.Z)

        # Ploteamos los puntos (x,y) y (x_best, y_best)
        # Nota:
        #   -> "bo" es para los puntos aleatorios
        #   -> "go" es para el mejor punto encontrado
        self.__plot_point(self.x, self.y, "bo")
        self.__plot_point(self.x_best, self.y_best, "go")

        # Realizamos una pausa para que se vea el progreso del "mejor" punto (x_best, y_best)
        plt.pause(0.005)



    # Función que grafica un punto en la grafica 2D o contour, se llamará iterativamente
    def __plot_point(self, x, y, marker):
        """
        Plotea un punto en la grafica 2D (contour)
        -------
        Private
        -------

        Parameters:
            x (Numeric): Posición en "x"
            y (Numeric): Posición en "y"
            marker (string) ['go'| "bo]: El tipo de marcador a plotear. Azul o verde 

        Returns: 
            None
        """

        # - El marcador verde es para el mejor punto encontrado por lo que se le agrega un label con el valor en "y"
        # - Como el marcador azul va a esta cambiando muy rápido entre iteraciones, no se lo coloca un label
        if marker == "go":
            label = "{:.2f}".format(y)

            # Anotamos el valor en "y" en la gráfica
            plt.annotate(
                label, (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
            )

        # Ploteamos el punto
        plt.plot(x, y, marker)







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
    plot_surface():
        Grafica la superficie en 3D de la función objetivo en el rango especificado
    __plot_contour():
        Grafica el contour de la función objetivo en el rango especificado
    __plot_point(x, y, marker):
        Plotea un punto en (x,y) con el color de marcador especificado
    search():
        Realiza el método de Gradiente descendente para calcular el mínimo global de la función objetivo.
        Además va graficando los valores encontrados y aproximaciones
    search_lite():
        Realiza el método de Gradiente descendente para calcular el mínimo global de la función objetivo.
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


        # Rango de valores entre el límite inferior y superior con una separación de 0.1
        # Se crea un rango tanto para "x" como para "y"
        self.x_range = np.arange(xl, xu, 0.1)
        self.y_range = np.arange(xl, xu, 0.1)

        # Nuestros vectores "x" y "y"        
        self.X, self.Y = np.meshgrid(self.x_range, self.y_range)

        # Calculamos nuestro vector "z" evaluando en la función objetivo "x" y "y"
        self.Z = self.f(self.X, self.Y)



    # Función para graficar la superficie en 3D de la función objetivo en el rango especificado
    def plot_surface(self):
        """
        Grafica la superficie en 3D de la función ebjetivo en el rango especificado

        Parameters:
            None

        Returns:
            None
        """

        # Se crea una ventana nueva para graficar.
        # Esto porque el contour puede ya estar en pantalla
        plt.figure()

        # Le decimos a la libreria de graficas que queremos la proyección en 3D
        ax = plt.axes(projection="3d")

        # Graficamos
        ax.plot_surface(self.X, self.Y, self.Z)

        # Se ponen títulos a los ejes
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        # Mostramos la gráfica
        plt.show()


    

    # - Función que grafica el contour de la función objetivo en el rango especificado,
    #   el punto inicial y actual obtenido en search() 
    # - Esta función será llamada iterativamente
    def __plot_contour(self):
        """
        Grafica la funcion objetivo en el rango especificado, el punto incial y el "mejor"
        -------
        Private
        -------

        Parameters:
            None
        
        Returns:    
            None
        """

        # Limpiamos el previo marcador que se había ploteado en la iteración anterior
        plt.clf()

        # Graficamos nuestro contour con los vectores "x", "y" y "z"
        plt.contour(self.X, self.Y, self.Z)

        # Ploteamos los puntos (x0,y0) o inicial y (xi, yi) o mejor valor encontrado
        # Nota:
        #   -> "bo" es para el punto inicial
        #   -> "go" es para el mejor punto encontrado
        self.__plot_point(self.x0, self.y0, "bo")
        self.__plot_point(self.xi, self.yi, "go")

        # Realizamos una pausa para que se vea el progreso del "mejor" punto (xi, yi)
        plt.pause(0.005)



    # Función que grafica un punto en la grafica 2D o contour, se llamará iterativamente    
    def __plot_point(self, x, y, marker):
        """
        Plotea un punto en la grafica 2D (contour)
        -------
        Private
        -------

        Parameters:
            x (Numeric): Posición en "x"
            y (Numeric): Posición en "y"
            marker (string) ['go'| "bo]: El tipo de marcador a plotear. Azul o verde 

        Returns: 
            None
        """

        # - El marcador verde es para el mejor punto encontrado por lo que se le agrega un label con el valor en "y"
        # - Como el marcador azul siempre será el mismo se coloca de color azul
        if marker == "go":
            label = "{:.2f}".format(y)

            # Anotamos el valor en "y" en la gráfica
            plt.annotate(
                label, (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
            )

        # Ploteamos el punto
        plt.plot(x, y, marker)



    # Función que realiza el método de Grtadiente Descendente para encontrar el mínimo global en
    # la función objetivo, además de ir graficando tanto el punto inicial como el
    # mejor punto encontrado hasta el momento
    def search(self, x0, y0, grad_func_x, grad_func_y):
        """
        Se realiza la búsqueda del mínimo global mediante Gradiente Descendente 
        y se van graficando los resultados 

        Parameters:
            x0 (Numeric): Valor incial en "x" donde es posible que se encuentre el mínimo global
            y0 (Numeric): Valor inicial en "y" donde es posible que se encuntre el mínimo global
            grad_func_x (Function): Función que calcula el gradiente para "xi". Es la derivada parcial
                de la función objetivo con respecto a "x"
            grad_func_y (Function): Función que calcula el gradiente para "yi". Es la derivada parcial
                de la función objetivo con respecto a "y"
        
        Returns: 
            None
        """
        self.x0 = x0
        self.y0 = y0
        self.xi = self.x0
        self.yi = self.y0

        # - Se itera calculando un mejor valor en cada iteración en base al anterior. 
        # - Entre más iteraciones se hagan más probable es encontrar un resultado más cercano al mínimo global.
        # - Se coloca un número pequeño de iteraciones porque al graficar y poder ver 
        #   el punto más acercado al resultado se necesitan realizar pequeñas pausas.
        for i in range(100):

            # - Se calcula un (x,y) en base al anterior (xi,yi). 
            # - Este servirá para la siguiente iteración
            # - Aquí se manda llamar a la función de cálculo de gradiente
            x_next = self.xi - self.h * grad_func_x(self.xi, self.yi)
            y_next = self.yi - self.h * grad_func_y(self.xi, self.yi)

            # Se actualiza el valor actual como el calculado
            self.xi = x_next
            self.yi = y_next

            print("Ploteando {},{}".format(self.xi, self.yi))

            # Graficamos el contour de la función. Dentro de esta se grafica también el valor
            # incial y el mejor valor hasta el momento
            self.__plot_contour()

        print(
            "Mínimo global en x={}, y={}, f(x)={}".format(
                round(self.xi, 2), round(self.yi, 2), round(self.f(self.xi, self.yi), 2)
            )
        )




    # Esta función realiza lo mismo que search(), exceptuando que en esta no se grafica, solo se obtiene
    # el valor del mínimo global. Por esto se puede colocar muchas más iteraciones
    def search_lite(self, x0, y0, grad_func_x, grad_func_y):
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
            None
        """


        self.x0 = x0
        self.y0 = y0
        self.xi = self.x0
        self.yi = self.y0

        # - Se itera calculando un mejor valor en cada iteración en base al anterior. 
        # - Entre más iteraciones se hagan más probable es encontrar un resultado más cercano al mínimo global.
        # - Se coloca un número pequeño de iteraciones porque al graficar y poder ver 
        #   el punto más acercado al resultado se necesitan realizar pequeñas pausas.
        for i in range(10000):

            # - Se calcula un (x,y) en base al anterior (xi,yi). 
            # - Este servirá para la siguiente iteración
            # - Aquí se manda llamar a la función de cálculo de gradiente
            x_next = self.xi - self.h * grad_func_x(self.xi, self.yi)
            y_next = self.yi - self.h * grad_func_y(self.xi, self.yi)

            # Se actualiza el valor actual como el calculado
            self.xi = x_next
            self.yi = y_next

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
#gradient = DescGradient(f1, xl=-2, xu=2, yl=-2, yu=2)
#gradient.search(-0.5,0, get_gradient_f1_x, get_gradient_f1_y)
#gradient.plot_surface()


# Segunda funcion
# gradient = DescGradient(f2, xl=1, xu=2, yl=1, yu=2 )
# gradient.search(1.8,1.8,get_gradient_f2_x, get_gradient_f2_y)
# gradient.plot_surface()
