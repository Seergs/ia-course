import matplotlib.pyplot as plt
import numpy as np
import unittest

def f(x1,x2):
    return (x1 - 2)**2 + (x2 - 2)**2


class GA(unittest.TestCase):
    def __init__(self, f, pop_size=100, dimension=2, xl=-5, xu=5, yl=-5, yu=5):

        self.f=f
        self.pop_size=pop_size
        self.dimension=dimension
        self.xl=xl
        self.xu=xu;
        self.yl=yl
        self.yu=yu
        self.x = np.empty((dimension, pop_size))
        self.fitness={}

    def plot_contour(self):
        x_range = np.arange(self.xl, self.xu, 0.1)
        y_range = np.arange(self.yl, self.yu, 0.1)

        # Nuestros vectores "x" y "y"
        X, Y = np.meshgrid(x_range, y_range)

        # Calculamos nuestro vector "z" evaluando en la función objetivo "x" y "y"
        Z = self.f(X, Y)
        plt.contour(X, Y, Z)
        plt.show()

    def start(self, generations=1000):
        print("\n\nIniciando simulación..\nEsto puede llevar bastante tiempo\n")
        xl_vector = np.array([self.xl,self.yl])
        xu_vector = np.array([self.xu,self.yu])

        
        self._initialize(xl_vector, xu_vector)

        for i in range(generations):
            self._evaluation()
            childs = self._create_childs_by_selecting_parents()
            self._mutate(childs, 0.01,xl_vector,xu_vector)
            self.x = childs

        key = max(self.fitness)
        print("Mínimo global en: {}".format(self.x[:,key]))

        


    def _initialize(self, xl_vector, xu_vector):
        for i in range(self.pop_size):
            self.x[:,i] = xl_vector + (xu_vector-xl_vector) * np.array((np.random.random(), np.random.random()))


    def _evaluation(self):
        for i in range(self.pop_size):
            fx = self.f(self.x[0,i], self.x[1,i])

            if fx >= 0:
                self.fitness[i] = 1/(1+fx)
            else:
                self.fitness[i] = 1+abs(fx)

            


    def _create_childs_by_selecting_parents(self):
        childs = np.empty((self.dimension, self.pop_size))
        for i in range(0,self.pop_size, 2):
            first_parent = self._get_index_by_roulette_wheel_selection()
        
            second_parent = first_parent

            while second_parent == first_parent:
                second_parent = self._get_index_by_roulette_wheel_selection()
            

            first_child, second_child = self._mix(self.x[:,first_parent], self.x[:,second_parent])

            childs[:,i] = first_child
            childs[:,i+1] = second_child

        
        return childs


    def  _mix(self, first_parent, second_parent):
        D = len(first_parent)
        pc = np.random.randint(0, D)

        first_child = first_parent
        second_child = second_parent

        first_child[pc:] = second_parent[pc:]
        second_child[pc:] = first_parent[pc:]

        return first_child, second_child

    
    def _mutate(self,y, pm, xl, xu):
        for i in range(self.pop_size):
            for j in range(self.dimension):
                if np.random.random() < pm:
                    y[j,i] = xl[j]+(xu[j]-xl[j]) * np.random.random()



    def _get_index_by_roulette_wheel_selection(self):
        total_fitness = sum(self.fitness.values())
        

        r = np.random.random()
        p_sum = 0

        for i in range(self.pop_size):
            p_sum += self.fitness[i]/total_fitness

            if p_sum >= r:
                return i
        
        return N


ga = GA(f, pop_size=200)
ga.start(generations=1000)
