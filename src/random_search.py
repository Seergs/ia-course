import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import math
from random import random
from mpl_toolkits import mplot3d
import time


def f1(x, y):
    return x * (np.e ** (-(x ** 2) - y ** 2))


def f2(x1, x2):
    return (x1 - 2) ** 2 + (x2 - 2) ** 2


def get_gradient(value):
    return 2 * (value - 2)


class RandomSearch:
    def __init__(self, f, xl=-10, xu=10, yl=-10, yu=10):
        self.f = f
        self.x = 0
        self.y = 0
        self.x_best = 0
        self.y_best = 0
        self.f_best = np.inf
        self.xl = xl
        self.xu = xu
        self.yl = yl
        self.yu = yu

        self.x_range = np.arange(xl, xu, 0.1)
        self.y_range = np.arange(xl, xu, 0.1)
        self.X, self.Y = np.meshgrid(self.x_range, self.y_range)
        self.Z = self.f(self.X, self.Y)

    def search(self):
        for i in range(10):
            self.x = self.xl + (self.xu - self.xl) * random()
            self.y = self.yl + (self.yu - self.yl) * random()

            f_val = self.f(self.x, self.y)
            if f_val < self.f_best:
                self.f_best = f_val
                self.x_best = self.x
                self.y_best = self.y

            self.__plot_contour()

            print("Ploteando {}, {}".format(self.x, self.y))

        print(
            "Mínimo global en x={}, y={}, f(x)={}".format(
                self.x_best, self.y_best, self.f_best
            )
        )
        self.__plot_surface()
        plt.show()

    def search_lite(self):
        for i in range(1000000):
            self.x = self.xl + (self.xu - self.xl) * random()
            self.y = self.yl + (self.yu - self.yl) * random()

            f_val = self.f(self.x, self.y)
            if f_val < self.f_best:
                self.f_best = f_val
                self.x_best = self.x
                self.y_best = self.y

        print(
            "Mínimo global en x={}, y={}, f(x)={}".format(
                round(self.x_best, 2), round(self.y_best, 2), round(self.f_best, 2)
            )
        )

    def __plot_contour(self):
        plt.clf()
        plt.contour(self.X, self.Y, self.Z)

        self.__plot_point(self.x, self.y, "bo")
        self.__plot_point(self.x_best, self.y_best, "go")
        plt.pause(0.005)

    def __plot_surface(self):
        plt.figure()
        mycmap = plt.get_cmap("gist_earth")
        ax = plt.axes(projection="3d")
        ax.plot_surface(self.X, self.Y, self.Z, cmap=mycmap)
        ax.plot(self.x_best, self.y_best, self.f_best, "ro", markersize="12")
        ax.view_init(elev=8, azim=-14)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    def __plot_point(self, x, y, marker):
        if marker == "go":
            label = "{:.2f}".format(y)
            plt.annotate(
                label, (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
            )
        plt.plot(x, y, marker)


class DescGradient:
    def __init__(self, f, gradient_func, x0, y0, xl=-10, xu=10, yl=-10, yu=10):
        self.f = f
        self.xl = xl
        self.xu = xu
        self.yl = yl
        self.yu = yu
        self.x0 = x0
        self.y0 = y0
        self.h = 0.1
        self.xi = self.x0
        self.yi = self.y0
        self.gradient_func = gradient_func

        self.x_range = np.arange(xl, xu, 0.1)
        self.y_range = np.arange(xl, xu, 0.1)
        self.X, self.Y = np.meshgrid(self.x_range, self.y_range)
        self.Z = self.f(self.X, self.Y)

    def plot(self):
        plt.figure()
        ax = plt.axes(projection="3d")
        ax.plot_surface(self.X, self.Y, self.Z)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()

    def search(self):
        for i in range(100):
            self.x_next = self.xi - self.h * get_gradient(self.xi)
            self.y_next = self.yi - self.h * get_gradient(self.yi)
            self.xi = self.x_next
            self.yi = self.y_next

        print(
            "Mínimo global en x={}, y={}, f(x)={}".format(
                round(self.xi, 2), round(self.yi, 2), round(self.f(self.xi, self.yi), 2)
            )
        )


# random_search = RandomSearch(f2, xl=1, xu=2, yl=1, yu=2)
# random_search.search()

gradient = DescGradient(f2, get_gradient, x0=1.8, y0=1.8, xl=1, xu=2, yl=1, yu=2)
gradient.search()
