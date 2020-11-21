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
#           Este programa calcula el mínimo global de una función con la estrategia evolutivas:
#           - (μ + λ)-ES
#       
#
#
##############################################################################################################

# Librerías necesarias para el correcto funcionamiento del programa
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.io import loadmat



# Clase principal encargada de realizar la estrategia evolutiva
# Para una comprensión más clara del algoritmo y de las variables
# buscar Estrategias evolutivas
class ES:

    # Método ejecutado al instanciar la clase, recibe la función objetivo,
    # el número de generaciones, los límites y el número de individuos de la población
    def __init__(self, f, xu, xl, generations, mu, lambda_, dimension, version=1):

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

    # Método que realiza la simulación de la estrategia evolutiva mediante
    # 1. Selección aleatoria de padres
    # 2. Recombinación de los padres para generar un hijo
    # 3. Ir ajustando mediante la suma de un vector "r"
    # 4. Los mejores individuos se convierten en los nuevos padres
    def calculate(self):

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
                    np.array(
                        (
                            self._x[0, self._mu + j],
                            self._x[1, self._mu + j],
                            self._x[2, self._mu + j],
                            self._x[3, self._mu + j],
                            self._x[4, self._mu + j],
                        )
                    )
                )

            # Se seleccionan los individuos que serán la nueva población
            self._x, self._sigma, self._fitness = self._select_new_population()

        w = self._x[:, 0]
        return w

    # Método que selecciona los mejores individuos de la población
    def _select_new_population(self):

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

        if select_parents:
            index = np.where(self._fitness == np.min(self._fitness[0]))[1][0]
        else:
            index = np.where(self._fitness == np.min(self._fitness[0, 30:]))[1][0]
        return index

    # Método que crea un vector aleatorio
    def _create_random_vector(self):
        # Se usa una distribución normal
        vector = np.random.normal(0, self._sigma[:, self._mu + 1])

        return vector

    # Método que combinas a dos padres para crear un hijo
    def _combine_parents(self, parent_1, parent_2):

        child = 0.5 * (parent_1 + parent_2)

        return child

    # Método que selecciona dos padres aleatorios
    def _select_parents(self):
        parent_1 = random.randint(0, self._mu)

        parent_2 = parent_1

        # Padres deben ser diferentes
        while parent_2 == parent_1:
            parent_2 = random.randint(0, self._mu)

        return parent_1, parent_2

    # Método que inicializa la población
    def _initialize_population(self):
        for i in range(self._mu):
            self._x[:, i] = self._create_random_individual()
            self._sigma[:, i] = self._create_random_sigma()
            self._fitness[0, i] = self._f(
                np.array(
                    (
                        self._x[0, i],
                        self._x[1, i],
                        self._x[2, i],
                        self._x[3, i],
                        self._x[4, i],
                    )
                )
            )

    # Método que crea un individuo con valores aleatorios
    def _create_random_individual(self):

        new_individual = self._xl + (self._xu - self._xl) * np.array(
            (
                np.random.random(),
                np.random.random(),
                np.random.random(),
                np.random.random(),
                np.random.random(),
            )
        )
        return new_individual

    # Método que crea una varianza aleatoria positiva
    def _create_random_sigma(self):

        new_sigma = 0.5 * np.array(
            (
                np.random.random(),
                np.random.random(),
                np.random.random(),
                np.random.random(),
                np.random.random(),
            )
        )

        return new_sigma


# Cargamos los datos para nuestra regresión no lineal
data = loadmat("src/exercise_4/exercise_4.mat")

# El tamaño del vector "x" de los datos
n = len(data["X"])

# Separamos los datos en "x" y "y"
X = data["X"]
Y = data["Y"]

# Polinomio
def g(w):
    result = w[0] + w[1] * X + w[2] * X ** 2 + w[3] * X ** 3 + w[4] * X ** 4

    return result


# Función objetivo 1
def f(w):

    z = (0.5 / n) * sum((Y - g(w)) ** 2)

    return z


es = ES(
    f=f,
    xu=np.array((10, 5, 5, 1, 0.1)),
    xl=np.array((-10, -5, -5, -1, -0.1)),
    generations=100,
    mu=30,
    lambda_=50,
    dimension=5,
    version=1,
)

w = es.calculate()
print(
    "Mínimo global en: w={},{},{},{},{}".format(
        round(w[0], 3), round(w[1], 3), round(w[2], 3), round(w[3], 3), round(w[4], 3)
    )
)
print("f(w)={}".format(f(w)))


# Ploteamos
plt.title("Regresión no lineal")
plt.plot(X, Y, "ro", label="Muestras")
plt.plot(X, g(w), "b-", label="Regresión")
plt.legend(loc="upper left")
plt.grid()
plt.show()
