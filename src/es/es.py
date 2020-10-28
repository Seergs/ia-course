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
    """
    def __init__(self, f, xu, xl, generations, mu, lambda_, dimension, version=1):
        self._f = f
        self._xu = xu
        self._xl = xl
        self._generations = generation
        self._mu = mu
        self._lambda = lambda_
        self._dimension = dimension
        self._version = version


        self._x = np.zeros((self._dimension, self._mu+self._lambda))
        self._sigma = np.zeros((self._dimension, self._mu+self._lambda))
        self._fitness = np.zeros((1, self._mu+self._lambda))


        x_range = np.arange(self._xl[0], self._xu[0], 0.1)
        y_range = np.arange(self._xl[1], self._xu[1], 0.1)

        # Nuestros vectores "x" y "y"
        self.X, self.Y = np.meshgrid(x_range, y_range)

        # Calculamos nuestro vector "z" evaluando en la función objetivo "x" y "y"
        self.Z = self._f(self.X, self.Y)

    def calculate(self):
        self._initialize_population()

        for i in range(self._generations):
            for j in range(self._lambda):
                r1, r2 = self._select_parents()

                self._x[:, self._mu+j] = self._combine_parents(r1,r2)
                self._sigma[:, self._mu+j]=self._combine_parents(self._sigma[:,r1], self._sigma[:,r2])

                r = self._create_random_vector()
                self._x[:,self._mu+j] += r

                self._fitness[0,self._mu+j] = self._f(self._x[0, self._mu+j], self._x[1, self._mu+j])

            
            self._plot_contour(i)


            self._x, self._sigma, self._fitness = self._select_new_population()

        plt.show()

        return self._x[0,0], self._x[1,0], self._fitness[0,0]

    
    def _select_new_population(self):
        new_x = np.zeros((self._dimension, self._mu+self._lambda))
        new_fitness = np.zeros((1, self._mu+self._lambda))
        new_sigma = np.zeros((self._dimension, self._mu+self._lambda))

        for i in range(self._mu):
            index = self._select_index_by_fitness(select_parents=True if self._version == 1 else False)
            new_x[:,i] = self._x[:, index]
            new_sigma[:,i] = self._sigma[:,index]
            new_fitness[0,i] = self._fitness[0,index]
            self._fitness[0,index] = np.inf

        return new_x, new_sigma, new_fitness


    def _select_index_by_fitness(self, select_parents=True):
        if select_parents:
            index = np.where(self._fitness == np.min(self._fitness[0]))[1][0]
        else:
            index = np.where(self._fitness == np.min(self._fitness[0,30:]))[1][0]
        return index

    
    def _create_random_vector(self):
        vector = np.random.normal(0, self._sigma[:,self._mu+1])

        return vector

    def _combine_parents(self, parent_1, parent_2):
        child = 0.5 * (parent_1+parent_2)

        return child

    
    def _select_parents(self):
        parent_1 = random.randint(0,self._mu)

        parent_2 = parent_1

        while parent_2 == parent_1:
            parent_2 = random.randint(0,self._mu)
        
        return parent_1, parent_2



    def _initialize_population(self):
        for i in range(self._mu):
            self._x[:,i] = self._create_random_individual()
            self._sigma[:,i] = self._create_random_sigma()
            self._fitness[0,i] = self._f(self._x[0,i], self._x[1,i])

    
    def _create_random_individual(self):
        new_individual = self._xl + (self._xu - self._xl) * np.array((np.random.random(), np.random.random()))
        return new_individual

    def _create_random_sigma(self):
        new_sigma = 0.5 * np.array((np.random.random(), np.random.random()))

        return new_sigma

    def plot(self):
        plt.contour(self.X, self.Y, self.Z)
        plt.show()

    def _plot_contour(self, generation):

        plt.clf()

        plt.title("Generación {}".format(generation))

        # Ploteamos la gráfica
        plt.contour(self.X, self.Y, self.Z)

        for i in range(self._mu):
            self._plot_point(self._x[0,i], self._x[1,i])

        plt.xlabel("x")
        plt.ylabel("y")

        # Se usa una pausa para que se actualice la gráfica
        plt.pause(0.005)
    
    def _plot_point(self, x, y):
        plt.plot(x, y, "ro")



def f(x,y):
    z = (x-2)**2 + (y-2)**2
    return z


es = ES(f=f, xu=np.array((5,5)),xl=np.array((-5,-5)), generations=200, mu=30, lambda_=50, dimension=2, version=2)
#es.plot()
x,y,fx = es.calculate()
print(x,y,fx)