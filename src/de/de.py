import numpy as np
import matplotlib.pyplot as plt

class DE:
    def __init__(self, f, population_size, xl, xu, dimension=2, F=0.6, CR=0.9):
        self._f = f
        self._population_size = population_size
        self._xl = xl
        self._xu = xu
        self._dimension = dimension
        self._F = F
        self._CR = CR
    

        self._x = np.zeros((self._dimension, self._population_size))
        self._fitness = np.zeros((self._population_size))


        # Las siguientes líneas nos ayudan para graficar
        x_range = np.arange(self._xl[0], self._xu[0], 0.1)
        y_range = np.arange(self._xl[1], self._xu[1], 0.1)

        # Nuestros vectores "x" y "y"
        self.X, self.Y = np.meshgrid(x_range, y_range)

        # Calculamos nuestro vector "z" evaluando en la función objetivo "x" y "y"
        self.Z = self._f(self.X, self.Y)

    
    def plot(self, title):
        plt.title(title)
        plt.contour(self.X, self.Y, self.Z)
        plt.show()

    def start(self, generations, plot=True):
        self._initialize()

        for n in range(generations):

            for i in range(self._population_size):

                r1 = i
                while r1 == i:
                    r1 = self._select_random_index_from_population()

                r2 = r1
                while r2 == r1 or r2 == i:
                    r2 = self._select_random_index_from_population()
                
                r3 = r2
                while r3 == r1 or r3 == r2 or r3 == i:
                    r3 = self._select_random_index_from_population()
                    
                u, v = self._recombine(r1,r2,r3,i)

                self._selection(u, i)
            
            if plot:
                self._plot_contour(n)
        
        if plot: 
            plt.show()
        
        
        best_index = self._get_best_index_by_fitness()
        x = self._x[:,best_index]
        f_best = self._fitness[best_index]
        return x, f_best 

                


    def _initialize(self):
        for i in range(self._population_size):
            self._x[:,i] = self._xl+(self._xu-self._xl)*np.random.random((self._dimension))
            self._fitness[i] = self._f(self._x[0,i], self._x[1,i])

    def _select_random_index_from_population(self):
        index = np.random.randint(0, self._population_size)

        return index


    def _recombine(self, r1, r2, r3, index):
        u = np.zeros((self._dimension))
        v = self._x[:,r1] + self._F*(self._x[:,r2]-self._x[:,r3])

        for j in range(self._dimension):
            r = np.random.random()

            if r <= self._CR:
                u[j] = v[j]
            else:
                u[j] = self._x[j,index]

        return u,v
    
    def _selection(self, u, index):
        xfx = self._f(self._x[0,index], self._x[1,index])
        ufx = self._f(u[0], u[1])

        if ufx < xfx:
            self._x[0, index] = u[0]
            self._x[1, index] = u[1]
            self._fitness[index] = ufx

    def _get_best_index_by_fitness(self):
        index = np.where(self._fitness == np.amin(self._fitness))[0][0]
        return index

    def _plot_contour(self, generation):

        plt.clf()

        plt.title("Generación {}".format(generation))

        plt.contour(self.X, self.Y, self.Z)
        axes = plt.gca()
        axes.set_xlim([self._xl[0], self._xu[0]])
        axes.set_ylim([self._xl[1], self._xu[1]])

        plt.plot(self._x[0], self._x[1], "ro", label="Soluciones", markersize=11)

        # Ploteamos todas las partículas
        #for i in range(self._population_size):
            #self._plot_point(self._x[0, i], self._x[1, i]) 

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


xl = np.array((-5,-5))
xu = np.array((5,5))

de = DE(f=sphere, population_size=50, xl=xl, xu=xu)
x, f_best = de.start(200)
x1, x2 = x
print("\n\nMínimo global encontrado en:")
print("{}".format(x1))
print("{}".format(x2))
print("{}".format(f_best))
