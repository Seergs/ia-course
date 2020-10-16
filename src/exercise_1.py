from newton import Newton

# Ejercicio 1
newton = Newton(
    value_range=[0,10],
    f= lambda l: l * (20-2*l)**2,
    fp= lambda l: 12*l**2 - 160 * l + 400,
    fpp = lambda l: 24*l - 160
)
newton.calculate([3])