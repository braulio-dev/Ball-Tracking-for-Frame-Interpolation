import numpy as np
from sympy import *
import math

FPS = None
PIXEL_POR_METRO = None


def PixelesAMetros(pixeles):
    return pixeles / PIXEL_POR_METRO

def MetrosAPixeles(metros):
    return metros * PIXEL_POR_METRO


# TODO: hacer que la velocidad sea calculada en metros por segundo y no en pixeles por frame
def VelocidadAngulo(x1, y1, x2, y2, cuadros):
    # Calculo de la velocidad y el angulo
    # Velocidad
    vx = (x2 - x1) / cuadros
    vy = (y2 - y1) / cuadros * -1  # por -1 porque la y crece hacia abajo
    v = math.sqrt(vx ** 2 + vy ** 2)
    # Angulo
    angulo = math.atan2(vy, vx) * 180 / math.pi

    v = PixelesAMetros(v) * FPS
    return angulo, v


def Punto(angulo, vi, distancia_x):
    distancia_x = PixelesAMetros(distancia_x)
    cant_terminos = 5
    x = Symbol('x')
    f = tan(angulo * pi / 180) * x + -9.81 * (x ** 2) / (2 * cos(angulo * pi / 180) * (vi ** 2))
    xi = distancia_x - 1
    xi1 = distancia_x
    h = xi1 - xi
    ValorReal = lambdify(x, f, 'numpy')(xi1)

    # Derivación
    df = [None] * cant_terminos
    df[0] = f

    for i in range(1, cant_terminos):
        df[i] = df[i - 1].diff(x)

    # Evaluación derivadas
    dxi = lambdify(x, df, 'numpy')
    derivadas = dxi(xi)

    # Serie Taylor
    st = [0] * cant_terminos
    et = [0] * cant_terminos
    for i in range(0, cant_terminos):
        acumulador = (derivadas[i] * pow(h, i)) / math.factorial(i)
        if i > 0:
            acumulador += st[i - 1]
        st[i] = acumulador

        # Error
        et[i] = ValorReal - acumulador

    # Resultados
    return int(MetrosAPixeles(distancia_x)), int(MetrosAPixeles(st[-1]))  # (x, y)
