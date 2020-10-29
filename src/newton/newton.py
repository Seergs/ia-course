"""
Evalua una funcion matemática para encontrar los máximos y mínimos con aproximación del método de Newton

Classes:

    Newton

Variables:

    newton

"""

import matplotlib.pyplot as plt
import numpy as np
import math


class Newton:
    """
    Clase para calcular los máximos y mínimos de una función por el método de Newton


    Attributes
    ----------
    initial_values : list
        Valores iniciales donde posiblemete exista una raíz
    value_range: list
        El rango a graficar, limite inferior y superior
    f: function
        Función que evalua la funcion objetivo en un punto
    fp: function
        Función que evalua la derivada de la función objetivo en un punto
    fpp: function
        Función que evalua la segunda derivada de la función objetivo en un punto

    Methods
    -------
    get_zeros_from_ecuation():
        Calcula los ceros o raíces de la ecuación con el método de Newton
    calculate():
        Calcula si las raíces son máximos o mínimos
    plot(roots):
        Grafica la función objetivo con matplotlib
    """

    # Método ejecutado al instanciar la clase,
    # recibe valores iniciales, el rango para graficar, función objetivo,
    # y derivadas de función objetivo
    def __init__(self, value_range, f, fp, fpp):
        """
        Inicializa la clase con los parametros inciales, la función objetivo y sus derivadas

            Parameters:
                initial_values (list): Lista de valores iniciales para método de Newton
                value_range (list): Límite inferior y superior para graficar la función
                f (function): Función que retorne la función objetivo evaluada en un punto
                fp (function): Función que retorne la derivada de la función objetivo evaluada en un punto
                fpp (function): Función que retorne la segunda derivada de la función objetivo evaluada en un punto

            Returns:
                None

        """
        self.range = value_range
        self.f = f
        self.fp = fp
        self.fpp = fpp
        print("\nClase instanciada, lista para calcular los máximos y mínimos\n")

    def plot(self):
        """
        Grafica la función objetivo con matplotlib

            Parameters:
                None

            Returns:
                None
        """

        # Creamos un rango de valores entre el limite inferior y superior
        # Este será nuestra x
        x = np.arange(self.range[0], self.range[1], 0.1)

        # Nuestra y la calculamos evaluando la función por cada valor de x
        y = []

        for value in x:
            y.append(self.f(value))

        # Ploteamos la función y los puntos críticos
        plt.plot(x, y)

        plt.title("Newton")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.show()

    # Funcion para realizar el método newton para aproximar las
    # raices de la ecuación
    def get_zeros_from_ecuation(self):
        """
        Calcula los ceros o raíces de la ecuación con el método de Newton

            Parameters:
                None

            Returns:
                roots (list): Lista con todas las raíces
        """

        roots = []

        # Por cada valor inicial (las posibles raíces) interpolamos
        for value in self.initial_values:
            x_i = value
            for i in range(10):
                x_next = x_i - (self.fp(x_i) / self.fpp(x_i))
                x_i = x_next
            roots.append(x_i)
            print("Raiz encontrada en: x={}".format(round(x_i, 2)))

        print("\n")
        return roots

    # Verifica en la segunda derivada, para saber si hay máximo
    # o minimo en las raices e imprime el resultado
    def calculate(self, initial_values):
        """
        Calcula si las raíces son máximos o mínimos

            Parameters:
               initial_values (list): Valores donde posiblemente exista una raíz

            Returns:
                None
        """
        self.initial_values = initial_values

        # Obtenemos las raices
        roots = self.get_zeros_from_ecuation()

        print("----Puntos críticos encontrados----")
        for x_ri in roots:

            # Si al evaluar la función el resultado es positivo, tenemos un mínimo,
            # si obtenemos un valor negativo, tenemos un máximo
            evaluated = self.fpp(x_ri)
            if evaluated > 0:
                print(
                    "\tMÍNIMO en: x={}, f(x)={}".format(
                        round(x_ri, 2), round(self.f(x_ri), 2)
                    )
                )
            else:
                print(
                    "\tMÁXIMO en: x={}, f(x)={}".format(
                        round(x_ri, 2), round(self.f(x_ri), 2)
                    )
                )

        self.__plot_results(roots)

    # Función para graficar la función objetivo
    def __plot_results(self, roots):
        """
        Grafica la función objetivo asíc como los puntos máximos y mínimos

            Parameters:
                roots (list): Lista de raices de la función

            Returns:
                None
        """

        # Creamos un rango de valores entre el limite inferior y superior
        # Este será nuestra x
        x = np.arange(self.range[0], self.range[1], 0.1)

        # Nuestra y la calculamos evaluando la función por cada valor de x
        y = []

        for value in x:
            y.append(self.f(value))

        # Ploteamos la función y los puntos críticos
        plt.plot(x, y)

        for root in roots:
            plt.plot(root, self.f(root), color="red", marker="o", markersize=12)

        plt.title("Newton")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.show()


# Primera funcion, le pasamos las derivadas como lambda functions
newton = Newton(
    value_range=[-4, 1],
    f=lambda x: x ** 4 + 5 * x ** 3 + 4 * x ** 2 - 4 * x + 1,
    fp=lambda x: 4 * x ** 3 + 15 * x ** 2 + 8 * x - 4,
    fpp=lambda x: 12 * x ** 2 + 30 * x + 8,
)
newton.calculate([-3, -1, 0.5])


# Segunda funcion, le pasamos las derivadas como lambda functions
# newton = Newton(value_range=[-4,4],
#   f=lambda x: math.sin(2*x),
#   fp=lambda x: 2*math.cos(2*x),
#   fpp=lambda x: -4*math.sin(2*x)
# )
# newton.calculate([-4,-2.5, -1, 1 ,2.5,4])

# Tercera funcion, le pasamos las derivadas como lambda functions
# newton = Newton(
#    value_range=[-5, 5],
#    f=lambda x: math.sin(x) + x * math.cos(x),
#    fp=lambda x: 2 * math.cos(x) - x * math.sin(x),
#    fpp=lambda x: -3 * math.sin(x) - x * math.cos(x),
# )
# newton.calculate([-3.9, -1,1,3.9])
