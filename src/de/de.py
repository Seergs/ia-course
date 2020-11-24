#################################################################################################################
#                                                                                                               #
#                                                                                                               #
#               Optimización por evolución diferencial                                                          #
#               Sergio Suárez Álvarez                                                                           #
#               Código 217758497                                                                                #
#               Seminario de solución de problemas de Inteligencia Artificial I                                 #
#                                                                                                               #
#               Este programa calcula el mínimo global de una función median evolución diferencial              #
#                                                                                                               #
#                                                                                                               #
#################################################################################################################


# Librerías necesarias para el correcto funcionamiento del programa
import numpy as np
import matplotlib.pyplot as plt


# Clase pincipal encargada de realizar la optimización de solución del mínimo global
# de una función por evolución diferencial
class DE:
    """
    Clase que realiza la optimización de una función mediante evolución diferencial

    Attributes
    ----------
    f: function
        Función objetivo
    population_size: integer
        Tamaño de la población
    xl: ndarray
        Vector de límites inferiores
    xu: ndarray
        Vector de límites superiores
    dimension: integer
        Dimensión del problema
    F: float
        Factor de amplificación [0,2]
    CR: float
        Constante de recombinación
    _x: ndarray
        Matriz que contiene a la población
    _fitness: ndarray
        Vector que contiene las aptitudes de las partículas
    X: array
        Lista de una dimensión que representa coordenadas de una matriz, para graficar el vector "x"
    Y: array
        Lista de una dimensión que representa coordenadas de una matriz, para graficar el vector "y"
    Z: array
        Lista de una dimensión que representa coordenadas de una matriz, para graficar el vector "z"



    Methods
    -------
    start(generations, plot):
        Función principal, realiza la optimización
    plot(title):
        Plotea la función
    _initialize():
        Inicializa la población con individuos aleatorios
    _select_random_index_from_population():
        Obtiene un índice aleatorio en el rango [0,dimension]
    _recombine(r1, r2, r3, index):
        Recombina los individuos para crear el vector de prueba
    _selection(u, index):
        Realiza la selección de individuos por aptitud
    _get_best_index_by_fitness():
        Obtiene el índice del elemento con mejor aptitud de la población
    _plot_contour(generation):
        Plotea la función y los individuos de la población en cada generación
    """

    # Método ejecutado al instanciar la clase, recibe la función objetivo, el tamaño de población,
    # y los límites superiores e inferiores, así como las contantes necesarias
    def __init__(self, f, population_size, xl, xu, dimension=2, F=0.6, CR=0.9):
        """
        Inicializa la clase

            Parameters:
                f (function): Función que retorna el valor al evaluar la función objetivo en un punto
                population_size (integer): Tamaño de la población
                xl (ndarray): Límites inferiores
                xu (ndarray): Límites superiores
                dimension (integer): Dimensión del problema
                F (float): Factor de amplificación [0,2]
                CR (float): Constante de recombinación

            Returns:
                None
        """
        self._f = f
        self._population_size = population_size
        self._xl = xl
        self._xu = xu
        self._dimension = dimension
        self._F = F
        self._CR = CR

        # Inicialización de matrices
        self._x = np.zeros((self._dimension, self._population_size))
        self._fitness = np.zeros((self._population_size))

        # Las siguientes líneas nos ayudan para graficar
        x_range = np.arange(self._xl[0], self._xu[0], 0.1)
        y_range = np.arange(self._xl[1], self._xu[1], 0.1)

        # Nuestros vectores "x" y "y"
        self.X, self.Y = np.meshgrid(x_range, y_range)

        # Calculamos nuestro vector "z" evaluando en la función objetivo "x" y "y"
        self.Z = self._f(self.X, self.Y)

    # Método que plotea el contour de la función objetivo
    def plot(self, title):
        """
        Plotea el contour de la función objetivo

            Parameters:
                title (string): Título a mostrar en la gráfica

            Returns:
                None
        """

        plt.title(title)
        plt.contour(self.X, self.Y, self.Z)
        plt.show()

    # Método que realiza la optimización por evolución diferencial
    def start(self, generations, plot=True):
        """
        Realiza la optimización por evolución diferencial

            Parameters:
                generations (integer): Número de generaciones
                plot (boolean): Si se desea plotear

            Returns:
                x (ndarray): Vector con la mejor solución
                f_best: (numeric): Aptitud de la mejor solución (x)
        """

        # 1. Inicialización
        self._initialize()

        for n in range(generations):

            for i in range(self._population_size):

                # Selección de 3 individuos diferentes aleatorios de la población
                r1 = i
                while r1 == i:
                    r1 = self._select_random_index_from_population()

                r2 = r1
                while r2 == r1 or r2 == i:
                    r2 = self._select_random_index_from_population()

                r3 = r2
                while r3 == r1 or r3 == r2 or r3 == i:
                    r3 = self._select_random_index_from_population()

                # 2. Recombinación
                u = self._recombine(r1, r2, r3, i)

                # 3. Selección
                self._selection(u, i)

            if plot:
                self._plot_contour(n)

        if plot:
            plt.show()

        best_index = self._get_best_index_by_fitness()
        x = self._x[:, best_index]
        f_best = self._fitness[best_index]
        return x, f_best

    # Método que inicializa a la población
    def _initialize(self):
        """
        Inicializa la población con individuos aleatorios

            Parameters:
                None

            Returns:
                None
        """

        # Creamos individuos aleatorios y su aptitud
        for i in range(self._population_size):
            self._x[:, i] = self._xl + (self._xu - self._xl) * np.random.random(
                (self._dimension)
            )
            self._fitness[i] = self._f(self._x[0, i], self._x[1, i])

    # Método que selecciona un índice aleatorio de la población
    def _select_random_index_from_population(self):
        """
        Obtiene un índice entre [0,dimension]

            Parameters:
                None

            Return:
                index (integer): Índice aleatorio
        """

        index = np.random.randint(0, self._population_size)

        return index

    # Método que recombina y muta a un individuo
    def _recombine(self, r1, r2, r3, index):
        """
        Muta y recombina a un individuo de la población

            Parameters
                r1 (integer): Índice del primer individuo aleatorio
                r2 (integer): Índice del segundo individuo aleatorio
                r3 (integer): Índice del tercer individuo aleatorio

            Returns:
                u (ndarray): Vector de prueba
        """
        u = np.zeros((self._dimension))

        # Creamos el vector mutante
        v = self._x[:, r1] + self._F * (self._x[:, r2] - self._x[:, r3])

        # Recombinación
        for j in range(self._dimension):
            r = np.random.random()

            if r <= self._CR:
                u[j] = v[j]
            else:
                u[j] = self._x[j, index]

        return u

    # Método que realiza la selección de los individuos con mejor aptitud
    def _selection(self, u, index):
        """
        Selecciona a los individuos con mejor aptitud

            Parameters:
                u (ndarray): Vector de mutación
                index (integer): Índice del individuo actual siendo validado

            Returns:
                None
        """

        # Evaluaciones en la función
        xfx = self._f(self._x[0, index], self._x[1, index])
        ufx = self._f(u[0], u[1])

        # Mejorando las aptitudes
        if ufx < xfx:
            self._x[0, index] = u[0]
            self._x[1, index] = u[1]
            self._fitness[index] = ufx

    # Método que obtiene el índice del individuo con mejor aptitud
    def _get_best_index_by_fitness(self):
        """
        Retorna el índice del individuo con mejor aptitud

            Parameters:
                None

            Returns:
                index (integer): Índice del individuo con mejor aptitud
        """

        index = np.where(self._fitness == np.amin(self._fitness))[0][0]
        return index

    # Método que plotea el contour y los individuos
    def _plot_contour(self, generation):
        """
        Plotea el contour de la función y los individuos por cada generación

            Parameters:
                generation: La generación actual

            Returns:
                None
        """
        plt.clf()

        plt.title("Generación {}".format(generation))

        plt.contour(self.X, self.Y, self.Z)
        axes = plt.gca()
        axes.set_xlim([self._xl[0], self._xu[0]])
        axes.set_ylim([self._xl[1], self._xu[1]])

        plt.plot(self._x[0], self._x[1], "ro", label="Soluciones", markersize=11)

        # Ploteamos todas las partículas
        # for i in range(self._population_size):
        # self._plot_point(self._x[0, i], self._x[1, i])

        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(loc="upper left")
        plt.grid()
        plt.pause(0.005)


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


xl = np.array((-5, -5))
xu = np.array((5, 5))

de = DE(f=sphere, population_size=50, xl=xl, xu=xu)
x, f_best = de.start(200)
x1, x2 = x
print("\n\nMínimo global encontrado en:")
print("{}".format(x1))
print("{}".format(x2))
print("{}".format(f_best))
