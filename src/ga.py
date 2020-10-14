import matplotlib.pyplot as plt
import numpy as np
import operator


def f(x1, x2):
    return (x1 - 2) ** 2 + (x2 - 2) ** 2


class GA:
    def __init__(self, f, pop_size=100, dimension=2, xl=-5, xu=5, yl=-5, yu=5):

        self.f = f
        self.pop_size = pop_size
        self.dimension = dimension
        self.xl = xl
        self.xu = xu
        self.yl = yl
        self.yu = yu
        self.x = np.empty((dimension, pop_size))
        self.fitness = {}

        x_range = np.arange(self.xl, self.xu, 0.1)
        y_range = np.arange(self.yl, self.yu, 0.1)
        self.X, self.Y = np.meshgrid(x_range, y_range)
        self.Z = self.f(self.X, self.Y)

    def plot(self):

        # Nuestros vectores "x" y "y"

        # Calculamos nuestro vector "z" evaluando en la función objetivo "x" y "y"
        plt.contour(self.X, self.Y, self.Z)
        plt.show()

    def start(self, generations=1000):
        print("\n\nIniciando simulación..\nEsto puede llevar bastante tiempo\n")
        xl_vector = np.array([self.xl, self.yl])
        xu_vector = np.array([self.xu, self.yu])

        self._initialize(xl_vector, xu_vector)

        for i in range(generations):
            self._evaluation()
            childs = self._create_childs_by_selecting_parents()
            self._mutate(childs, 0.01, xl_vector, xu_vector)
            self.x = childs
            self._plot_contour(i)

        plt.show()

        x_best, y_best = self._get_best_fitness()
        print(
            "Mínimo global en x={}, y={}, f(x)={}\n".format(
                round(x_best, 2), round(y_best, 2), round(self.f(x_best, y_best), 2)
            )
        )

    def _initialize(self, xl_vector, xu_vector):
        for i in range(self.pop_size):
            self.x[:, i] = xl_vector + (xu_vector - xl_vector) * np.array(
                (np.random.random(), np.random.random())
            )

    def _evaluation(self):
        for i in range(self.pop_size):
            fx = self.f(self.x[0, i], self.x[1, i])

            if fx >= 0:
                self.fitness[i] = 1 / (1 + fx)
            else:
                self.fitness[i] = 1 + abs(fx)

    def _create_childs_by_selecting_parents(self):
        childs = np.empty((self.dimension, self.pop_size))
        for i in range(0, self.pop_size, 2):
            first_parent = self._get_index_by_roulette_wheel_selection()

            second_parent = first_parent

            while second_parent == first_parent:
                second_parent = self._get_index_by_roulette_wheel_selection()

            first_child, second_child = self._mix(
                self.x[:, first_parent], self.x[:, second_parent]
            )

            childs[:, i] = first_child
            childs[:, i + 1] = second_child

        return childs

    def _mix(self, first_parent, second_parent):
        D = len(first_parent)
        pc = np.random.randint(0, D)

        first_child = first_parent
        second_child = second_parent

        first_child[pc:] = second_parent[pc:]
        second_child[pc:] = first_parent[pc:]

        return first_child, second_child

    def _mutate(self, y, pm, xl, xu):
        for i in range(self.pop_size):
            for j in range(self.dimension):
                if np.random.random() < pm:
                    y[j, i] = xl[j] + (xu[j] - xl[j]) * np.random.random()

    def _get_index_by_roulette_wheel_selection(self):
        total_fitness = sum(self.fitness.values())

        r = np.random.random()
        p_sum = 0

        for i in range(self.pop_size):
            p_sum += self.fitness[i] / total_fitness

            if p_sum >= r:
                return i

        return N

    def _get_best_fitness(self):
        key = max(self.fitness.items(), key=operator.itemgetter(1))[0]

        return self.x[0, key], self.x[1, key]

    def _plot_contour(self, i):
        plt.clf()

        plt.contour(self.X, self.Y, self.Z)

        x, y = self._get_best_fitness()
        self._plot_point(x, y, i)
        plt.pause(0.005)

    def _plot_point(self, x, y, i):
        plt.title("Generación {}".format(i))
        label = "{:.2f},{:.2f}".format(x, y)
        plt.annotate(
            label, (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
        )
        plt.plot(x, y, "ro")


ga = GA(f, pop_size=200)
# ga.plot()
ga.start(generations=20)
