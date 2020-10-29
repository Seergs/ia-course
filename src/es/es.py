###############################################################################################################
#
#
#
#           Estrategias evolutivas
#           Sergio Suárez Álvarez
#           Código 217758497
#           Seminario de solución de problemas de inteligencia Artificial I
#
#
#           Este programa calcula el mínimo global de una función con las estrategias evolutivas:
#           - (μ + λ)-ES
#           - (μ, λ)-ES
#
#
##############################################################################################################

# Librerías necesarias para el correcto funcionamiento del programa
import numpy as np
import random
import matplotlib.pyplot as plt


"""
    Encuentra el mínimo global de una función

    Classes:

        ES
    
    Variables:

        es

"""


# Clase principal encargada de realizar las estrategias evolutivas
# Para una comprensión más clara del algoritmo y de las variables
# buscar Estrategias evolutivas
class ES:
    """
    Clase que ejecuta las estrategias evolutivas para obtener
    el mínimo global de una función


    Attributes
    ----------
    f: function
        Función objetivo a la cual se le calculará el mínimo global
    generations: integer
        El número de generaciones a evolucionar
    xu: ndarray
        Límites superiores de la función
    xl: ndarray
        Límites inferiores de la función
    mu: integer
        Número de padres de la población
    lambda_: integer
        Número de hijos de la población
    dimension: integer
        Dimensión de la función
    version: integer (1|2)
        Contrala la versión a utilizar: 1 = (μ + λ)-ES, 2 = (μ, λ)-ES
    _x: ndarray
        Matriz que contiene la población
    _sigma: ndarray
        Matriz que contiene las varianzas estándar de la distribución de probabilidad de la población
    _fitness: ndarray
        Vector que contiene las aptitutes de la población
    X: array
        Lista de una dimensión que representa coordenadas de una matriz, para graficar el vector "x"
    Y: array
        Lista de una dimensión que representa coordenadas de una matriz, para graficar el vector "y"
    Z: array
        Lista de una dimensión que representa coordenadas de una matriz, para graficar el vector "z"


    Methods
    -------
    calculate():
        Calcula el mínimo global de la función
    plot():
        Grafica el contour la función
    _select_new_population():
        Selecciona a los nuevos individuos para la siguiente generación
    _select_index_by_fitness(select_parents):
        Retorna el individuo con mejor aptitud
    _create_random_vector():
        Genera un vector aleatorio que se le sumará a los individuos
    _combine_parents(parent_1, parent_2):
        Crea un hijo en base a los padres
    _select_parents():
        Selecciona dos padres aleatorios
    _initialize_population():
        Inicializa los individuos de la población
    _create_random_individual():
        Crea una tupla con la información aleatoria para un individuo
    _create_random_sigma():
        Crea una varianza estándar aleatoria positiva para un individuo
    _plot_contour(generation):
        Grafica una generación en el contour de la función
    _plot_point(x,y):
        Grafica un punto en "x" y "y" en el contour de la función
    """

    # Método ejecutado al instanciar la clase, recibe la función objetivo,
    # el número de generaciones, los límites y el número de individuos de la población
    def __init__(self, f, xu, xl, generations, mu, lambda_, dimension, version=1):
        """
        Inicializa la clase

            Parameters:
                f (Function): Función que retorna el valor al evaluar la función objetivo en un punto
                xu (ndarray): Vector con los límites superiores de la función
                xl (ndarray): Vector con los límites inferiores de la función
                generations (integer): Número de generaciones para la simulación
                mu (integer): Número de padres de la población
                lambda_ (integer): Número de hijos de la población
                dimension (integer): Dimensión de la función
                version (integer): Versión del algoritmo a realizar. 1 = (μ + λ)-ES, 2 = (μ, λ)-ES

            Returns:
                None
        """

        self._f = f
        self._xu = xu
        self._xl = xl
        self._generations = generations
        self._mu = mu
        self._lambda = lambda_
        self._dimension = dimension
        self._version = version

        # Inicializamos nuestras matrices
        self._x = np.zeros((self._dimension, self._mu + self._lambda))
        self._sigma = np.zeros((self._dimension, self._mu + self._lambda))
        self._fitness = np.zeros((1, self._mu + self._lambda))

        # Las siguientes líneas nos ayudan para graficar
        x_range = np.arange(self._xl[0], self._xu[0], 0.1)
        y_range = np.arange(self._xl[1], self._xu[1], 0.1)

        # Nuestros vectores "x" y "y"
        self.X, self.Y = np.meshgrid(x_range, y_range)

        # Calculamos nuestro vector "z" evaluando en la función objetivo "x" y "y"
        self.Z = self._f(self.X, self.Y)

    # Método que realiza la simulación de la estrategia evolutiva mediante
    # 1. Selección aleatoria de padres
    # 2. Recombinación de los padres para generar un hijo
    # 3. Ir ajustando mediante la suma de un vector "r"
    # 4. Los mejores individuos se convierten en los nuevos padres
    def calculate(self):
        """
        Realiza la simulación de la estrategia evolutiva

            Parameters:
                None

            Returns:
                x (Numeric): Valor en "x" donde se encuentra el mínimo global
                y (Numeric): Valor en "y" donde se encuentra el mínimo global
                z (Numeric): Valor en "z" donde se encuentra el mínimo global
        """

        # Inicializamos la población aleatoria
        self._initialize_population()

        for i in range(self._generations):
            for j in range(self._lambda):

                # Selección de padres
                r1, r2 = self._select_parents()

                # Recombinación de dos padres para crear un hijo
                self._x[:, self._mu + j] = self._combine_parents(r1, r2)
                self._sigma[:, self._mu + j] = self._combine_parents(
                    self._sigma[:, r1], self._sigma[:, r2]
                )

                # Ajuste mediante el vector
                r = self._create_random_vector()
                self._x[:, self._mu + j] += r

                # Cálculo de la aptitud
                self._fitness[0, self._mu + j] = self._f(
                    self._x[0, self._mu + j], self._x[1, self._mu + j]
                )

            # self._plot_contour(i)

            # Se seleccionan los individuos que serán la nueva población
            self._x, self._sigma, self._fitness = self._select_new_population()

        # plt.show()

        x = self._x[0, 0]
        y = self._x[1, 0]
        z = self._fitness[0, 0]
        return x, y, z

    # Método que selecciona los mejores individuos de la población
    def _select_new_population(self):
        """
        Selecciona los mejores individuos de la población

            Parameters:
                None

            Returns:
                new_x (ndarray): Matriz con la nueva población
                new_sigma (ndarray): Matriz con las nuevas varianzas estándar
                new_fitness (ndarray): Vector con las nuevas aptitudes
        """

        new_x = np.zeros((self._dimension, self._mu + self._lambda))
        new_fitness = np.zeros((1, self._mu + self._lambda))
        new_sigma = np.zeros((self._dimension, self._mu + self._lambda))

        for i in range(self._mu):

            # Dependiendo de la versión del algoritmo se eligen entre todos lo invididuos o solo los hijos
            select_parents = True if self._version == 1 else False
            index = self._select_index_by_fitness(select_parents=select_parents)

            new_x[:, i] = self._x[:, index]
            new_sigma[:, i] = self._sigma[:, index]
            new_fitness[0, i] = self._fitness[0, index]
            self._fitness[0, index] = np.inf

        return new_x, new_sigma, new_fitness

    # Método que obtiene el índice del mejor individuo de la población
    def _select_index_by_fitness(self, select_parents=True):
        """
        Retorna el índice del individuo con mejor aptitud de la población

            Parametes:
                select_parents (Boolean): Define si la búsqueda será solo en los hijos o también en los padres

            Returns:
                index (integer): Índice del mejor individuo de la población
        """

        if select_parents:
            index = np.where(self._fitness == np.min(self._fitness[0]))[1][0]
        else:
            index = np.where(self._fitness == np.min(self._fitness[0, 30:]))[1][0]
        return index

    # Método que crea un vector aleatorio
    def _create_random_vector(self):
        """
        Crea un vector aleatorio

            Parameters:
                None

            Returns:
                vector (ndarray): Vector aleatorio
        """

        # Se usa una distribución normal
        vector = np.random.normal(0, self._sigma[:, self._mu + 1])

        return vector

    # Método que combinas a dos padres para crear un hijo
    def _combine_parents(self, parent_1, parent_2):
        """
        Recombina dos padres y creas un hijo

            Paramenters:
                parent_1 (ndarray): Padre 1
                parent_2 (ndarray): Padre 2

            Returns:
                child (ndarray): Hijo creado por la recombinación de los padres

        """

        child = 0.5 * (parent_1 + parent_2)

        return child

    # Método que selecciona dos padres aleatorios
    def _select_parents(self):
        """
        Selecciona dos padres aleatorios

            Parameters:
                None

            Returns:
                parent_1: Padre 1
                parent_2: Padre 2
        """

        parent_1 = random.randint(0, self._mu)

        parent_2 = parent_1

        # Padres deben ser diferentes
        while parent_2 == parent_1:
            parent_2 = random.randint(0, self._mu)

        return parent_1, parent_2

    # Método que inicializa la población
    def _initialize_population(self):
        """
        Inicializa la población con valores aleatorios

            Parameters:
                None

            Returns:
                None
        """

        for i in range(self._mu):
            self._x[:, i] = self._create_random_individual()
            self._sigma[:, i] = self._create_random_sigma()
            self._fitness[0, i] = self._f(self._x[0, i], self._x[1, i])

    # Método que crea un individuo con valores aleatorios
    def _create_random_individual(self):
        """
        Crea un individuo con valores aleatorios

            Parameters:
                None

            Returns:
                new_individual (ndarray): El nuevo individuo
        """

        new_individual = self._xl + (self._xu - self._xl) * np.array(
            (np.random.random(), np.random.random())
        )
        return new_individual

    # Método que crea una varianza aleatoria positiva
    def _create_random_sigma(self):
        """
        Crea una varianza aleatoria positiva

            Parameters:
                None

            Returns:
                new_sigma (ndarray): La nueva varianza aleatoria positiva
        """

        new_sigma = 0.5 * np.array((np.random.random(), np.random.random()))

        return new_sigma

    # Método que plotea el contour de la gráfica
    def plot(self, title):
        """
        Plotea el contour de la gráfica

            Parameters:
                title (string): Título de la gráfica

            Returns:
                None
        """
        plt.title(title)
        plt.contour(self.X, self.Y, self.Z)
        plt.show()

    # Método que plotea el contour de la función y la mejor solución de la generación
    def _plot_contour(self, generation):
        """
        Plotea el contour de la función y la mejor solución por generación

            Parameters:
                generation (integer): La generación actual

            Returns:
                None
        """

        plt.clf()

        plt.title("Generación {}".format(generation))

        plt.contour(self.X, self.Y, self.Z)

        self._plot_point(self._x[0, 0], self._x[1, 0])

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

        label = "{:.2f},{:.2f}".format(x, y)
        plt.annotate(
            label, (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
        )
        plt.plot(x, y, "ro")


# Función objetivo 1
def f_1(x, y):
    """
    Evalúa la función objetivo en "x" y "y"

        Parameters:
            x (Numeric): Valor en "x" a evaluar
            y (Numeric): Valor en "y" a evaluar

        Returns:
            z (Numeric): Valor de la función evaluada en (x,y)
    """

    z = x * np.e ** (-(x ** 2) - y ** 2)

    return z


# Función objetivo 2
def f_2(x, y):
    """
    Evalúa la función objetivo en "x" y "y"

        Parameters:
            x (Numeric): Valor en "x" a evaluar
            y (Numeric): Valor en "y" a evaluar

        Returns:
            z (Numeric): Valor de la función evaluada en (x,y)
    """

    z = (x - 2) ** 2 + (y - 2) ** 2

    return z


# Instanciación de la clase con la función objetivo 1
# es = ES(
#    f=f_1,
#    xu=np.array((2, 2)),
#    xl=np.array((-2, -2)),
#    generations=200,
#    mu=30,
#    lambda_=50,
#    dimension=2,
#    version=1,
# )


# Instanciación de la clase con la función objetivo 2
es = ES(
    f=f_2,
    xu=np.array((5, 5)),
    xl=np.array((-5, -5)),
    generations=200,
    mu=30,
    lambda_=50,
    dimension=2,
    version=1,
)


# es.plot("Resultado")
x, y, fx = es.calculate()
print(
    "\n\nMínimo global en x={}, y={}, f(x)={}\n\n".format(
        round(x, 3), round(y, 3), round(fx, 3)
    )
)
