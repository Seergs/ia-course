import matplotlib.pyplot as plt
import numpy as np

class PSO:
    def __init__(self, f, xl, xu, swarm_size, dimension):
        self._f = f
        self._xl = xl
        self._xu = xu
        self._swarm_size = swarm_size
        self._dimension = dimension
        self._w = 0.6
        self._c1 = 2
        self._c2 = 2
        
        self._x = np.zeros((self._dimension, self._swarm_size))
        self._xb = np.zeros((self._dimension, self._swarm_size))
        self._v = np.zeros((self._dimension, self._swarm_size))
        self._fitness = np.zeros((1, self._swarm_size))

        x_range = np.arange(self._xl[0], self._xu[0], 0.1)
        y_range = np.arange(self._xl[1], self._xu[1], 0.1)

        # Nuestros vectores "x" y "y"
        self.X, self.Y = np.meshgrid(x_range, y_range)

        # Calculamos nuestro vector "z" evaluando en la función objetivo "x" y "y"
        self.Z = self._f(self.X, self.Y)

    def simulate(self, generations, plot=False):
        self._initialize()

        for j in range(generations):
            for i in range(self._swarm_size):
                fx = self._f(self._x[0,i], self._x[1,i])

                if fx < self._fitness[0,i]:
                    self._xb[:,i] = self._x[:,i]
                    self._fitness[0,i] = fx
            
            index = self._select_best_index_by_fitness()

            for i in range(self._swarm_size):
                first_factor = self._w*self._v[:,i]
                second_factor = self._c1*np.random.random()*(self._xb[:,i]-self._x[:,i])
                third_factor = self._c2*np.random.random()*(self._xb[:,index]-self._x[:,i])

                self._v[:,i] = first_factor + second_factor + third_factor
                self._x[:,i] = self._x[:,i] + self._v[:,i]

            if plot:
                self._plot_contour(j)

        
        if plot:
            plt.show()

        best = self._select_best_index_by_fitness()
        x_best = self._x[0,best]
        y_best = self._x[1,best]
        f_best = self._fitness[0,best]
        return x_best, y_best, f_best
    

    def _initialize(self):
        for i in range(self._swarm_size):
            self._x[:,i] = self._xl + (self._xu - self._xl) * np.array(
                (np.random.random(), np.random.random())
                )
            self._xb[:,i] = self._x[:,i]
            self._v[:,i] = 0.5* np.array((np.random.uniform(-1,1), np.random.uniform(-1,1)))
            self._fitness[0,i] = self._f(self._x[0,i], self._x[1,i])


    def _select_best_index_by_fitness(self):
        
        index = np.where(self._fitness == np.min(self._fitness[0]))[1][0]
        return index

    def _plot_contour(self, generation):

        plt.clf()

        plt.title("Generación {}".format(generation))

        plt.contour(self.X, self.Y, self.Z)
        axes = plt.gca()
        axes.set_xlim([self._xl[0], self._xu[0]])
        axes.set_ylim([self._xl[1], self._xu[1]])

        for i in range(self._swarm_size):
            self._plot_point(self._x[0, i], self._x[1, i])

        plt.xlabel("x")
        plt.ylabel("y")

        plt.pause(0.005)

    def _plot_point(self, x, y):
        plt.plot(x, y, "ro")
        

    
    def plot(self, title):
        plt.title(title)
        plt.contour(self.X, self.Y, self.Z)
        plt.show()


def sphere(x1,x2):
    z = x1**2+x2**2

    return z

def griewank(x1,x2):
    sumatory = (x1**2/4000)+(x2**2/4000)
    product = np.cos(x1/np.sqrt(1))*np.cos(x2/np.sqrt(2))

    z = sumatory - product + 1

    return z

def rastrigin(x1,x2):
    sumatory = (x1**2 - 10*np.cos(2*np.pi*x1)) + (x2**2 - 10*np.cos(2*np.pi*x2))
    z = 20 + sumatory


    return z

pso = PSO(f=sphere, xl=np.array((-5,-5)), xu=np.array((5,5)), swarm_size=50, dimension=2)
#pso = PSO(f=griewank, xl=np.array((-5,-5)), xu=np.array((5,5)), swarm_size=50, dimension=2)
#pso = PSO(f=rastrigin, xl=np.array((-5,-5)), xu=((5,5)), swarm_size=50, dimension=2)
x_best, y_best, f_best = pso.simulate(50, plot=True)
print("Mínimo global encontrado en x={}, y={}, f(x)={}".format(x_best, y_best, f_best))