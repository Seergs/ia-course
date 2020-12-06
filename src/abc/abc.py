############################################################################################
#                                                                                          #
#                                                                                          #
#                  Optimización por colonia de abejas artificial                           #
#                  Sergio Suárez Álvarez                                                   #
#                  Código 21775849                                                         #
#                  Seminario de solución de problemas de Inteligencia Artificial           #
#                                                                                          #
#                  Este programa calcula el mínimo global de una función mediante          #
#                  colonia de abejas                                                       #
#                                                                                          #
############################################################################################

# Librerías necesarias para el correcto funcionamiento del programa
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


# Clase principal encargada de realizar la optimización de una función
# por colonia de abejas
class ABC:
    """
    Realiza la optimización de una función mediante colonia de abejas

    Attributes
    ----------
    f: function
        Función objetivo
    xl: ndarray
        Vector de límites inferiores
    xu: ndarray
        Vector de límites superiores
    dimension: integer
        Dimensión del problema
    population_size: integer
        Tamaño de la población total (abejas trabajadoras + abejas observadoras)
    L: integer
        Número máximo de intentos de una abeja trabajdora de explotar una solución
    Pf: integer
        Número de abejas trabajadoras
    Po: integer
        Número de abejas observadoras
    x: ndarray
        Matriz con la población
    l: ndarray
        Vector con los números de intentos de cada individuo
    aptitude: ndarray
        Vector con las calidades de aptitud de los individuos
    fitness: ndarray
        Vector con las evaluaciones de las soluciones en la función objetivo
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
    plot_contour(title):
        Función que grafica el contour de la función
    plot_surface(title):
        Función que grafica la superficie de la función en 3D
    _initialize():
        Inicializa la población con individuos aleatorios
    _workers_phase():
        Realiza la fase de las abejas trabajadoras
    _observers_phase():
        Realiza la fase de las abejas observadoras
    _explorers_phase():
        Realiza la fase de las abejas exploradoras
    _get_random_index_from_population():
        Obtiene un índice aleatorio en el rango [0, Pf]
    _get_index_by_roulette_wheel_selection():
        Devuelve un índice mediante selección por ruleta
    _evaluation():
        Realiza la evalución de las soluciones en la función objetivo para obtener
        sus calidades de aptitud
    _get_best_index_by_fitness():
        Retorna el índice del individuo con mejor aptitud
    _plot(generations):
        Plotea la función y los individuos de la población en cada generación
    """

    # Método ejecutado al instanciar la clase, recibe la función objetivo, el tamaño de población,
    # y los límites superiores e inferiores, así como las contantes necesarias
    def __init__(self, f, xl, xu, dimension, population_size, L=30, Pf=30):
        """
        Inicializa la clase

            Parameters:
                f (function): Función que retorna el valor al evaluar la función objetivo en un punto
                xl (ndarray): Límites inferiores
                xu (ndarray): Límites superiores
                dimension (integer): Dimensión del problema
                population_size (integer): Tamaño de la población
                L (integer): Número máximo de intentos de explotar una solución por una abeja trabajadora
                Pf (integer): Número de abejas trabajadoras

            Returns:
                None
        """
        self._f = f
        self._xl = xl
        self._xu = xu
        self._dimension = dimension
        self._population_size = population_size
        self._L = L
        self._Pf = Pf
        # Calculamos el número de abejas observadoras
        self._Po = self._population_size - self._Pf

        # Creamos nuestros vectores y matrices
        self._x = np.zeros((self._dimension, self._Pf))
        self._l = np.zeros(self._Pf)
        self._aptitude = np.zeros(self._Pf)
        self._fitness = np.zeros(self._Pf)

        # Las siguientes líneas nos ayudan para graficar
        x_range = np.arange(self._xl[0], self._xu[0], 0.1)
        y_range = np.arange(self._xl[1], self._xu[1], 0.1)

        # Nuestros vectores "x" y "y"
        self._X, self._Y = np.meshgrid(x_range, y_range)

        # Calculamos nuestro vector "z" evaluando en la función objetivo "x" y "y"
        self._Z = self._f(self._X, self._Y)

    # Método que realiza la optimización por colonia de abejas
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

        # Fase de inicialización
        self._initialize()

        # Se realizan las tres fases en cada iteración
        for i in range(generations):
            self._workers_phase()
            self._observers_phase()
            self._explorers_phase()
            if plot:
                self._plot(i)

        if plot:
            plt.show()

        # Se obtiene el mejor
        best_index = self._get_best_index_by_fitness()
        return self._x[:, best_index], self._f(
            self._x[0, best_index], self._x[1, best_index]
        )

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
        for i in range(self._Pf):
            self._x[:, i] = self._xl + (self._xu - self._xl) * np.random.random(
                (self._dimension)
            )
            self._fitness[i] = self._f(self._x[0, i], self._x[1, i])

    # Etapa de abejas empleadas
    def _workers_phase(self):
        """
        Etapa de abejas trabajadoras, explotando las soluciones

            Parameters:
                None

            Returns:
                None
        """

        for i in range(self._Pf):
            k = i
            while k == i:
                k = self._get_random_index_from_population()

            j = np.random.randint(0, self._dimension)
            phi = 2 * np.random.random() - 1

            v = np.copy(self._x[:, i])

            # Calculando una nueva solución
            v[j] = self._x[j, i] + phi * (self._x[j, i] - self._x[j, k])

            fv = self._f(v[0], v[1])

            # Mejorando las soluciones
            if fv < self._fitness[i]:
                self._x[:, i] = v
                self._fitness[i] = fv

                # Si se encontró una mejor solución, se resetea el número de intentos
                self._l[i] = 0
            else:
                # Si no se encontró una mejor solución, se incrementa el número de intentos
                self._l[i] = self._l[i] + 1

        # Al final se hace la evaluación, para que en la fase de las observadoras se pueda hacer
        # use de la selección por ruleta
        self._evaluation()

    # Etapa de abejas observadoras
    def _observers_phase(self):
        """
        Etapa de abejas observadoras

            Parameters:
                None
            Returns:
                None
        """

        for i in range(self._Po):

            # Se obtiene un índice mediante selección por ruleta
            m = self._get_index_by_roulette_wheel_selection()

            # Se realiza lo mismo que la etpa de las trabajadoras pero ahora con el índice
            # obtenido mediante suleta (m)
            k = m
            while k == m:
                k = self._get_random_index_from_population()

            j = np.random.randint(0, self._dimension)
            phi = 2 * np.random.random() - 1

            v = np.copy(self._x[:, m])
            v[j] = self._x[j, m] + phi * (self._x[j, m] - self._x[j, k])

            fv = self._f(v[0], v[1])

            # Mejorando
            if fv < self._fitness[m]:
                self._x[:, m] = v
                self._fitness[m] = fv
                self._l[m] = 0
            else:
                self._l[m] = self._l[m] + 1

    # Etapa de abejas exploradoras
    def _explorers_phase(self):
        """
        Etapa de abejas exploradoras

            Parameters:
                None
            Returns:
                None
        """

        for i in range(self._Pf):

            # Si se agotaron los intentos de la abeja se crea una nueva solución
            if self._l[i] > self._L:

                self._x[:, i] = self._xl + (self._xu - self._xl) * np.random.random(
                    (self._dimension)
                )
                self._fitness[i] = self._f(self._x[0, i], self._x[1, i])
                self._l[i] = 0

    # Método que selecciona un índice aleatorio de la población
    def _get_random_index_from_population(self):
        """
        Obtiene un índice entre [0,Pf]

            Parameters:
                None

            Return:
                index (integer): Índice aleatorio
        """

        index = np.random.randint(0, self._Pf)

        return index

    # Función de utilidad para obtener un índice de los individuos de la población
    # utilizando el algoritmo de la selección por ruleta, que selecciona con mayor
    # probabilidad a los individuos con mejor aptitud
    def _get_index_by_roulette_wheel_selection(self):
        """
        Obtiene un índice de la matriz de los individuos mediante el algoritmo
        de la ruleta para seleccionar al individuos con mejor (probablemente) aptitud

            Parameters:
                None

            Returns:
                i (Numeric): Índice del individuo de la población
                /Pf (Numeric): Si no se encuentra un individuo se devuelve el último elemento
        """

        # Se obtiene la aptitud total
        total_fitness = sum(self._aptitude)

        r = np.random.random()
        p_sum = 0

        for i in range(self._Pf):
            p_sum += self._aptitude[i] / total_fitness

            if p_sum >= r:
                return i

        return self._Pf

    # Función que evalua cada individuo de la población, calculando su aptitud
    def _evaluation(self):
        """
        Evalúa los individuos de la población para calcular su aptitud

            Parameters:
                None

            Returns:
                None
        """

        for i in range(self._Pf):
            # Se evalúa el individuo en la función objetivo
            fx = self._f(self._x[0, i], self._x[1, i])

            # Se calcula su aptitud
            if fx >= 0:
                self._aptitude[i] = 1 / (1 + fx)
            else:
                self._aptitude[i] = 1 + abs(fx)

    # Método que obtiene el índice del individuo con mejor aptitud

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

    # Método que plotea el contour de la función objetivo
    def plot_contour(self, title):
        """
        Plotea el contour de la función objetivo

            Parameters:
                title (string): Título de la gráfica

            Returns:
                None
        """

        plt.title(title)
        plt.contour(self._X, self._Y, self._Z)
        plt.show()

    # Método que plotea la superficie de la función objetivo
    def plot_surface(self, title):
        """
        Plotea la superficie de la función objetivo

            Parameters:
                title (string): Título de la gráfica

            Returns:
                None
        """

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(self._X, self._Y, self._Z, cmap=cm.coolwarm)
        plt.title(title)
        plt.show()

    # Método que plotea la función objetivo y los individuos
    def _plot(self, generation):
        """
        Plotea el contour de la función y los individuos por cada generación

            Parameters:
                generation: La generación actual

            Returns:
                None
        """
        plt.clf()

        plt.title("Generación {}".format(generation))

        plt.contour(self._X, self._Y, self._Z)
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


xl = np.array((-5, -5))
xu = np.array((5, 5))

abc = ABC(f=griewank, xl=xl, xu=xu, dimension=2, population_size=50)
x, f_best = abc.start(200)
x1, x2 = x
print("\n\nMínimo global encontrado en:")
print("x={}".format(x1))
print("y={}".format(x2))
print("f(x)={}\n\n".format(f_best))
