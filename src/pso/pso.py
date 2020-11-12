#################################################################################################
#
#
#
#
#
#           Optimización por enjambre de particulas
#           Sergio Suárez Álvarez
#           Código 217759497
#           Seminario de solución de problemas de Inteligencia Artificial I
#
#
#           Este programa calcula el mínimo global de una función mediante enjambre de partículas
#
#
#
################################################################################################


# Librerías necesarias para el correcto funcionamiento del programa
import matplotlib.pyplot as plt
import numpy as np

"""
    Encuentra el mínimo global de una función

    Classes:

        PSO

    Variables:
        pso
"""


# Clase principal encargada de realizar la optimización de solución del
# mínimo global de una fuinción por enjambre de partículas
class PSO:
    """
    Clase que ejecuta la optimización por enjambre de partículas
    para encontrar el mínimo global de una función


    Attributes
    ----------
    f: function
        Función objetivo a la cual se le calculará el mínimo global
    xl: ndarray
        Límites inferiores de la función
    xu: ndarray
        Límites superiores de la función
    swarm_size: integer
        Tamaño del enjambre o población
    dimension: integer
        La dimensión del problema
    _w: numeric
        Factor de inercia, el cual provoca en la partícula mantener su
        dirección si este valor es grande
    _c1: numeric
        Factor de aprendizaje cognitivo
    _c2: numeric
        Factor de aprendizaje social
    _x: ndarray
        Matriz que contiene al enjambre
    _xb: ndarray
        Matriz que contiene las mejores posiciones de las partículas
    _v: ndarray
        Matriz con las velocidades de las partículas
    _fitness: ndarray
        Vector con las aptitudes de las partículas
     X: array
        Lista de una dimensión que representa coordenadas de una matriz, para graficar el vector "x"
    Y: array
        Lista de una dimensión que representa coordenadas de una matriz, para graficar el vector "y"
    Z: array
        Lista de una dimensión que representa coordenadas de una matriz, para graficar el vector "z"



    Methods
    -------
    _initialize():
        Inicializa al enjambre
    _select_best_index_by_fitness():
        Retorna el índice de la partícula con la mejor aptitud
    _plot_contour(generation):
        Plotea el contour de la función en cada generación
    _plot_point(x,y):
        Plotea un punto en el contour
    plot(title):
        Plotea el contour de la función
    simulate(generations, plot):
        Inicia la optimización por enjambre de partículas mediante el número de generaciones
    """

    # Método ejecutado al instanciar la clase, recibe la función objetivo,
    # el tamaño del enjambre y los límites de la función
    def __init__(self, f, xl, xu, swarm_size, dimension=2):
        """
        Inicializa la clase

            Parameters:
                f (function): Función que retorna el valor al evaluar la función objetivo en un punto
                xl (ndarray): Límites inferiores de la función
                xu (ndarray): Límites superiores de la función
                swarm_size (integer): Tamaño del enjambre
                dimension (integer): La dimensión del problema

            Returns:
                None
        """

        self._f = f
        self._xl = xl
        self._xu = xu
        self._swarm_size = swarm_size
        self._dimension = dimension

        # Constantes
        self._w = 0.6
        self._c1 = 2
        self._c2 = 2

        # Inicialización de matrices
        self._x = np.zeros((self._dimension, self._swarm_size))
        self._xb = np.zeros((self._dimension, self._swarm_size))
        self._v = np.zeros((self._dimension, self._swarm_size))
        self._fitness = np.zeros((1, self._swarm_size))

        # Las siguientes líneas nos ayudan para graficar
        x_range = np.arange(self._xl[0], self._xu[0], 0.1)
        y_range = np.arange(self._xl[1], self._xu[1], 0.1)

        # Nuestros vectores "x" y "y"
        self.X, self.Y = np.meshgrid(x_range, y_range)

        # Calculamos nuestro vector "z" evaluando en la función objetivo "x" y "y"
        self.Z = self._f(self.X, self.Y)

    # Función que ejecuta el algoritmo de optimización
    # por enjambre de partículas
    def simulate(self, generations, plot=False):
        """
        Realiza la optimización de una función para encontrar su mínimo global

            Parameters:
                generations (integer): Número de iteraciones a realizar
                plot (Boolean): Si se quiere plotear los resultados en cada generación


            Returns:
                x_best (numeric): Valor en "x" donde se encuentra el mínimo global
                y_best (numeric): Valor en "y" donde se encuentra el mínimo global
                f_best (numeric): Valor en "z" donde se encuentra el mínimo global
        """

        # Inicializamos al enjambre
        self._initialize()

        for j in range(generations):
            for i in range(self._swarm_size):
                fx = self._f(self._x[0, i], self._x[1, i])

                # Comprobamos si la evaluación de una partícula
                # es mejor que su mejor aptitud al momento
                if fx < self._fitness[0, i]:
                    self._xb[:, i] = self._x[:, i]
                    self._fitness[0, i] = fx

            # Obtenemos la partícula lider
            index = self._select_best_index_by_fitness()

            for i in range(self._swarm_size):

                # Fórmula de actualización del PSO
                first_factor = self._w * self._v[:, i]
                second_factor = (
                    self._c1 * np.random.random() * (self._xb[:, i] - self._x[:, i])
                )
                third_factor = (
                    self._c2 * np.random.random() * (self._xb[:, index] - self._x[:, i])
                )

                self._v[:, i] = first_factor + second_factor + third_factor
                self._x[:, i] = self._x[:, i] + self._v[:, i]

            if plot:
                self._plot_contour(j)

        if plot:
            plt.show()

        best = self._select_best_index_by_fitness()
        x_best = self._x[0, best]
        y_best = self._x[1, best]
        f_best = self._fitness[0, best]

        return x_best, y_best, f_best

    # Método que inicializa al enjambre
    def _initialize(self):
        """
        Inicializa al enjambre con valores aleatorios

            Parameters:
                None

            Returns:
                None
        """

        for i in range(self._swarm_size):
            self._x[:, i] = self._xl + (self._xu - self._xl) * np.array(
                (np.random.random(), np.random.random())
            )
            self._xb[:, i] = self._x[:, i]
            self._v[:, i] = 0.5 * np.array(
                (np.random.uniform(-1, 1), np.random.uniform(-1, 1))
            )
            self._fitness[0, i] = self._f(self._x[0, i], self._x[1, i])

    # Método que obtiene el índice de la mejor partícula del enjambre
    def _select_best_index_by_fitness(self):
        """
        Retorna el índice de la partícula con mejor aptitud

            Parameters:
                None

            Returns:
                index (integer): Índice de la partícula
        """

        index = np.where(self._fitness == np.min(self._fitness[0]))[1][0]
        return index

    # Método que plotea el contour de la función y las partículas de la generación
    def _plot_contour(self, generation):
        """
        Plotea el contour de la función y las partículas de la generación
            Parameters:
                generation (integer): La generación actual
            Returns:
                None
        """

        plt.clf()

        plt.title("Generación {}".format(generation))

        plt.contour(self.X, self.Y, self.Z)
        axes = plt.gca()
        axes.set_xlim([self._xl[0], self._xu[0]])
        axes.set_ylim([self._xl[1], self._xu[1]])

        # Ploteamos todas las partículas
        for i in range(self._swarm_size):
            self._plot_point(self._x[0, i], self._x[1, i])

        plt.xlabel("x")
        plt.ylabel("y")

        plt.pause(0.005)

    # Método que plotea un punto en el contour
    def _plot_point(self, x, y):
        """
        Plotea un punto en el contour de la gráfica
            Parameters:
                x (Numeric): Valor en "x" a plotear
                y (Numeric): Valor en "y" a plotear
            Returns:
                None
        """

        plt.plot(x, y, "ro")

    # Método que plotea el contour de la función
    def plot(self, title):
        """
        Plotea el contour de la función

            Parameters:
                title (string): Título de la gráfica

            Returns:
                None
        """

        plt.title(title)
        plt.contour(self.X, self.Y, self.Z)
        plt.show()


# Función objetivo 3
def sphere(x1, x2):
    """
    Función objetivo 3

        Parameters:
            x1 (numeric): Valor en "x" a evaluar
            x2 (numeric): Valor en "y" a evaluar

        Returns:
            z (numeric): Valor en "z" al evaluar la función
    """
    z = x1 ** 2 + x2 ** 2

    return z


# Función objetivo 1
def griewank(x1, x2):
    """
    Función objetivo 1

        Parameters:
            x1 (numeric): Valor en "x" a evaluar
            x2 (numeric): Valor en "y" a evaluar

        Returns:
            z (numeric): Valor en "z" al evaluar la función
    """
    sumatory = (x1 ** 2 / 4000) + (x2 ** 2 / 4000)
    product = np.cos(x1 / np.sqrt(1)) * np.cos(x2 / np.sqrt(2))

    z = sumatory - product + 1

    return z


# Función objetivo 2
def rastrigin(x1, x2):
    """
    Función objetivo 2

        Parameters:
            x1 (numeric): Valor en "x" a evaluar
            x2 (numeric): Valor en "y" a evaluar

        Returns:
            z (numeric): Valor en "z" al evaluar la función
    """
    sumatory = (x1 ** 2 - 10 * np.cos(2 * np.pi * x1)) + (
        x2 ** 2 - 10 * np.cos(2 * np.pi * x2)
    )
    z = 20 + sumatory

    return z


pso = PSO(f=sphere, xl=np.array((-5, -5)), xu=np.array((5, 5)), swarm_size=50)
x_best, y_best, f_best = pso.simulate(50, plot=True)
print("\n\nMínimo global encontrado\nx={}, y={}, f(x)={}\n\n".format(x_best, y_best, f_best))
