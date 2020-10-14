#   Sergio Suárez Álvarez
#   217758497
#   Seminario de solución de problemas de Inteligencia Artificial 1
#   
#   Este programa busca el mínimo global de una función
#   usando un algoritmo genético clásico.
#   Para más información sobre el GA, visite https://towardsdatascience.com/introduction-to-genetic-algorithms-including-example-code-e396e98d8bf3#:~:text=A%20genetic%20algorithm%20is%20a,offspring%20of%20the%20next%20generation
#
#
#

'''
    Encuentra el mínimo global de una función

    Classes:

        GA

    Variables:
        ga

'''


# Librerías necesarias para el correcto funcionamiento
# de nuestro programa
import matplotlib.pyplot as plt
import numpy as np
import operator


# Función objetivo 2
def f2(x1, x2):
    '''
    Evalúa la función objetivo en el punto x,y

        Parameters:
            x (Numeric): coordenada en el eje "x"
            y (Numeric): coordenada en el eje "y"

        Returns:
            z (Numeric): valor de la función en x,y

    '''

    # Evaluamos la función objetivo en x,y
    z = (x1 - 2) ** 2 + (x2 - 2) ** 2

    return z






# Clase para ejecutar el algoritmo genético
class GA:
    """
    Clase que ejecuta el algoritmo genético para obtener
    el mínimo global (aproximado) de una función


    Attributes
    ----------
    f: function
        Función objetivo a la cual se le calculará el mínimo global
    pop_size: Numeric
        El tamaño de la población de las posibles soluciones
    dimension: Numeric
        La dimensión o número de variables a calcular
    xl: Numeric
        Límite inferior en "x"
    xu: Numeric
        Límite superior en "x"
    yl: Numeric
        Límite inferior en "y"
    yu: Numeric
        Límite superior en "y"
    x: ndarray
        Matriz con la población de posibles soluciones
    fitness: dictionary
        Lista de aptitudes de la población
    X: array
        Lista de una dimensión que representa coordenadas de una matriz, para
        graficar el vector "x"
    Y: array
        Lista de una dimensión que representa coordenadas de una matriz, para
        graficar el vector "y"
    Z: array
        Lista de una dimensión que representa coordenadas de una matriz, para
        graficar el vector "z"

    
    Methods
    -------
    plot():
        Grafica la función en el rango especificado
    start(generations):
        Inicia la simulación hasta completar las generaciones
    _initialize(xl_vector, xu_vector):
        Inicializa a la población con valores aleatorios
    _evaluation():
        Evalúa cada individuo de la población para opbtener su aptitud
    _create_childs_by_selecting_parents():
        Itera por la población, seleccionando a dos padres para crear hijos
    _mix(first_parent, second_parent):
        Cruza entre dos padres para crear dos hijos, intercambiando el material genético
    _mutate(y, pm, xl, xu):
        Muta a los hijos con una probablidad baja
    """




    # Método ejecutado al instanciar la clase, recibe la función objetivo,
    # el tamaño de la población (con valor por defecto de 100), 
    # y los límites tanto superior como inferior de "x" y "y"
    def __init__(self, f, pop_size=100, xl=-5, xu=5, yl=-5, yu=5):
        """
        Inicializa la clase con el rango de valores, la función objetivo
        y el tamaño de la población

        Paramenters:
            f (function): Función que retorna el valor al evaluar la función objetivo en un punto
            pop_size (Numeric): Tamaño de la población
            xl (Numeric): Límite inferior en "x"
            xu (Numeric): Límite superior en "x"
            yl (Numeric): Límite inferior en "y"
            yu (Numeric): Límite superior en "y"

        Returns: 
            None
        """

        self.f = f
        self.pop_size = pop_size

        #Inicializamos la dimension en 2
        self.dimension = 2
        self.xl = xl
        self.xu = xu
        self.yl = yl
        self.yu = yu

        # Creamos una matriz de DxN vacía, en esta se colocará la población en
        # cada generación
        self.x = np.empty((self.dimension, pop_size))


        #Inicializamos la lista de aptitudes como vacía
        self.fitness = {}

        # Rango de valores entre el límite inferior y superior con una separación de 0.1
        # Se crea un rango tanto para "x" como para "y"
        x_range = np.arange(self.xl, self.xu, 0.1)
        y_range = np.arange(self.yl, self.yu, 0.1)



        # Nuestros vectores "x" y "y"
        self.X, self.Y = np.meshgrid(x_range, y_range)

        # Calculamos nuestro vector "z" evaluando en la función objetivo "x" y "y"
        self.Z = self.f(self.X, self.Y)







    # Función que grafica el contour de la función objetivo en el rango especificado
    def plot(self):
        """
        Grafica la función objetivo en el rango especificado

            Parameters:
                None
            
            Returns:
                None

        """

        # Se usa matplotlib para plotear nuestro contour        
        plt.contour(self.X, self.Y, self.Z)
        plt.show()





    # Función inicial que comenza con la simulación del algoritmo genético
    def start(self, generations=1000):
        """
        Utiliza el algoritmo genético para simular la evolución

            Parameters:
                generations (Numeric): Número de iteraciones (generaciones) a realizar

            Returns:
                None
        """



        print("\n\nIniciando simulación..\nEsto puede llevar bastante tiempo\n")


        # Creamos dos vectores con numpy para nuestros límites.
        # Se usa este tipo de vector y no un array porque numpy nos permita
        # realizar operaciones con ellos de una manera muy sencilla
        xl_vector = np.array([self.xl, self.yl])
        xu_vector = np.array([self.xu, self.yu])


        # 1. Llamamos a nuestro método encargado de inicializar la población
        self._initialize(xl_vector, xu_vector)


        # Iteramos en las generaciones
        for i in range(generations):

            # 2. Evaluamos los individuos de la población, calculando su aptitud
            self._evaluation()


            # 3. Se obtiene una matriz de hijos seleccionado a los padres mejor calificados
            childs = self._create_childs_by_selecting_parents()

            # 4. Se mutan los hijos, con una paqueña probabilidad
            self._mutate(childs, 0.01, xl_vector, xu_vector)

            # 5. Los hijos se convierten en la nueva población
            self.x = childs

            # Se grafica en el contour el punto con mejor aptitud
            self._plot_contour(i)
                

        # Esta línea nos permite tener la gráfica "abierta" mientras el algoritmo sigue corriendo
        plt.show()



        # Finalmente obtenemos nuestro vector con mejor aptitud
        x_best, y_best = self._get_best_fitness()
        print(
            "Mínimo global en x={}, y={}, f(x)={}\n".format(
                round(x_best, 2), round(y_best, 2), round(self.f(x_best, y_best), 2)
            )
        )




    # Función que inicializa la población
    def _initialize(self, xl_vector, xu_vector):
        """
        Inicializa la población

            Parameters:
                xl_vector (ndarray): Vector con los límites inferiores del rango
                xu_vector (ndarray): Vector con los límites superiores del rango

            Returns: 
                None
        """


        # Se itera creando individuos dentro del espacio de trabajo del espacio continuo
        for i in range(self.pop_size):
            self.x[:, i] = xl_vector + (xu_vector - xl_vector) * np.array(
                (np.random.random(), np.random.random())
            )




    # Función que evalua cada individuo de la población, calculando su aptitud
    def _evaluation(self):
        """
        Evalúa los individuos de la población para calcular su aptitud

            Parameters: 
                None

            Returns:
                None
        """

        for i in range(self.pop_size):
            # Se evalúa el individuo en la función objetivo
            fx = self.f(self.x[0, i], self.x[1, i])

            # Se calcula su aptitud
            if fx >= 0:
                self.fitness[i] = 1 / (1 + fx)
            else:
                self.fitness[i] = 1 + abs(fx)





    # Función que crea a los hijos en base a la selección de dos padres
    def _create_childs_by_selecting_parents(self):
        """
        Crea una nueva población de hijos seleccionado a los dos 
        padres aptos mediante ruleta

            Parameters:
                None

            Returns: 
                childs (ndarray): Matriz con los hijos creados
        """


        # Se crea una matriz vacía donde se colocarán los hijos
        childs = np.empty((self.dimension, self.pop_size))


        # Se itera de dos en dos en la población actual
        for i in range(0, self.pop_size, 2):


            # Se selecciona a los padres mediante ruleta
            first_parent = self._get_index_by_roulette_wheel_selection()

            # Se crea un segundo padre igual al primero para poder
            # realizar la comprobación de abajo
            second_parent = first_parent


            # El algortitmo dice que se seleccione a los padres n1 y n2 donde n1 != n2
            # por eso se realiza esta comprobación
            while second_parent == first_parent:
                second_parent = self._get_index_by_roulette_wheel_selection()


            # Se cruza a los padres para crear a los dos hijos 
            first_child, second_child = self._mix(
                self.x[:, first_parent], self.x[:, second_parent]
            )

            # Se agregan a la matriz
            childs[:, i] = first_child
            childs[:, i + 1] = second_child

        return childs





    # Función para cruzar a los padres y crear a los hijos
    def _mix(self, first_parent, second_parent):
        """
        Cruza a los padres y retorna los hijos

            Parameters:
                first_parent (vector): El primer padre
                second_parent (vector): El segundo padre 

            Returns:

        """

        # Se crea el punto de cruce
        D = len(first_parent)
        pc = np.random.randint(0, D)

        # Se les coloca el material genético
        first_child = first_parent
        second_child = second_parent

        # A partir del punto de cruce se les pasa la información respectivamente
        first_child[pc:] = second_parent[pc:]
        second_child[pc:] = first_parent[pc:]

        return first_child, second_child





    # Función que crea mutaciones en los hijos
    def _mutate(self, y, pm, xl, xu):
        """
        Muta a los hijos con una paqueña probabilidad

            Parameters:
                y (ndarray): La matriz de hijos
                pm (Numeric): Probabilidad de mutación.
                xl (vector): Vector de los límites inferiores del rango
                xu (vector): Vector de los límites superiores del rango

            Returns: 
                None
        """

        # Se itera por absolutamente todos los hijos
        for i in range(self.pop_size):
            for j in range(self.dimension):

                # Se crea un número aleatorio y si este es menor que la probabilidad de mutación
                # entonces este hijo se muta con la fórmula utilizada en la inicialización
                if np.random.random() < pm:
                    y[j, i] = xl[j] + (xu[j] - xl[j]) * np.random.random()






    # Función de utilidad para obtener un índice de los individuos de la población
    # utilizando el algoritmo de la selección por ruleta, que selecciona con mayor
    # probabilidad a los individuos con mejor aptitud
    def _get_index_by_roulette_wheel_selection(self):
        """
        Obtiene un índice de la matriz de los individuos mediante el algoritmo
        de la ruleta para seleccionar a los padres con mejor (probablemente) aptitud

            Parameters:
                None
            
            Returns:
                i (Numeric): Índice del individuo de la población
                /pop_size (Numeric): Si no se encuentra un individuo se devuelve el último elemento
        """

        # Se obtiene la aptitud total
        total_fitness = sum(self.fitness.values())

        r = np.random.random()
        p_sum = 0

        for i in range(self.pop_size):
            p_sum += self.fitness[i] / total_fitness

            if p_sum >= r:
                return i

        return self.pop_size





    # Función que retorn el elemento con mejor aptitud de la población
    def _get_best_fitness(self):
        """
        Retorna el individuo de la población con mejor aptitud

            Parameters:
                None

            Returns:
                x, y (tuple): El vector con mejor aptitud
        """

        # Como la aptitud es un diccionario se debe de realizar lo siguiente
        # para obtener el key que tenga un value más alto
        key = max(self.fitness.items(), key=operator.itemgetter(1))[0]

        x = self.x[0,key]
        y = self.x[1,key]

        return x,y





    # Función que plotea tanto la gráfica de la función objetivo 
    # como el punto con mejor aptitud en cada generación
    # Esta función se ejecuta en cada generación
    def _plot_contour(self, i):
        """
        Grafica la función objetivo y el punto con mejor aptitud de cada generación

            Parameters:
                i (Numeric): La generación actual

            Returns:
                None
            
        """


        # Limpíamos la gráfica para eliminar el marcador de la generación pasada
        plt.clf()

        # Ploteamos la gráfica
        plt.contour(self.X, self.Y, self.Z)

        # Obtenemos el mejor vector
        x, y = self._get_best_fitness()

        # Ploteamos el mejor vector
        self._plot_point(x, y, i)

        plt.xlabel("x")
        plt.ylabel("y")

        # Se usa una pausa para que se actualice la gráfica
        plt.pause(0.005)



    # Función que plotea un punto en la gráfica
    def _plot_point(self, x, y, i):
        """
        Plotea un solo (x,y) en la gráfica

            Parameters:
                x (Numeric): Coordenada en "x"
                y (Numeric): Coordenada en "y"
                i (Numeric): Generación actual

            Returns:
                None
        """

        # Se le coloca como título a la gráfica la generación actual
        plt.title("Generación {}".format(i))

        label = "{:.2f},{:.2f}".format(x, y)
        plt.annotate(
            label, (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
        )

        # Se plotea el punto
        plt.plot(x, y, "ro")







# Se instancia la clase con la función objetivo y una población de 200
ga = GA(f2, pop_size=200)

# ga.plot()

# Se inicia la simulación con 1000 generaciones
ga.start(generations=100)
