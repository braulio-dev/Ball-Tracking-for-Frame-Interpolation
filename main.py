import argparse
from time import time, sleep

import cv2 as cv
import imutils

import predicciones
from predicciones import *
from collections import deque

# Argumentos
args_parser = argparse.ArgumentParser()
args_parser.add_argument("-a", "--archivo", help="Archivo de entrada (opcional)", required=False)
args_parser.add_argument("-r", "--radio", help="Radio estimado, en metros, del objeto a seguir", required=True)
args_parser.add_argument("-m", "--metodo", help="Metodo de predicción a usar (opcional) (Taylor, Euler, Runge Kutta)", required=False)
args = vars(args_parser.parse_args())

# Declaración de variables
color_lower = (29, 86, 6)  # Amarillo verde
color_upper = (64, 255, 255)
buffer = 16
tamanio_horizontal = 700
radio = float(args["radio"])  # Radio del objeto a seguir
metodo = args["metodo"].lower().replace(" ", "_") if args["metodo"] else "taylor"
running = False
num_cuadro = 0
cuadro_previo = None
angulo_previo = None
posicion_inicial = None
puntos_iniciales = []
puntos_reales = deque(maxlen=buffer)
puntos_predichos = deque(maxlen=buffer)
velocidad_inicial = None
angulo_inicial = None
start_time = 0
cuadro_intermedio = None
posicion_intermedia = None
radioPix = 0

# Tomar el archivo de entrada y declarar la escala
captura = cv.VideoCapture(args["archivo"])
usar_prediccion = False
predicciones.FPS = fps = captura.get(cv.CAP_PROP_FPS)
cuadros_totales = captura.get(cv.CAP_PROP_FRAME_COUNT)
tiempo_esperado = cuadros_totales / fps


def restart(rebote):
    global running, angulo_previo, radioPix, posicion_intermedia, cuadro_intermedio, posicion_inicial, puntos_iniciales, puntos_reales, puntos_predichos, velocidad_inicial, angulo_inicial, num_cuadro, start_time, cuadro_previo, captura
    running = True
    posicion_inicial = None
    puntos_iniciales = []
    velocidad_inicial = None
    angulo_inicial = None
    cuadro_previo = None
    radioPix = 0
    angulo_previo = None
    cuadro_intermedio = None
    posicion_intermedia = None
    puntos_predichos = deque(maxlen=buffer)
    if not rebote:
        captura = cv.VideoCapture(args["archivo"])
        num_cuadro = 0
        start_time = time()
        puntos_reales = deque(maxlen=buffer)

def ultima_posicion():
    pos_1 = None
    pos_2 = None
    pos = 0
    pos2 = 0
    for i in range(0, len(puntos_reales)):
        pos = i
        if puntos_reales[i] is not None:
            pos_2 = puntos_reales[i]
            break
    for j in range(pos + 1, len(puntos_reales)):
        pos2 = j
        if puntos_reales[j] is not None:
            pos_1 = puntos_reales[j]
            break
    return pos_1, pos_2, (pos2 - pos) + 1


def ha_calculado():
    return velocidad_inicial and angulo_inicial


def interpolar(cuadro):
    global cuadro_intermedio
    cuadro_intermedio = cuadro.copy()
    x, y = posicion_intermedia
    cv.circle(cuadro_intermedio, (int(x), int(y)), int(radioPix), (0, 0, 0), 2)
    cv.circle(cuadro_intermedio, (int(x), int(y)), 5, (0, 0, 0), -1)


def tomar_cuadro_intermedio():
    global cuadro_intermedio, posicion_intermedia
    cuadro = cuadro_intermedio
    cuadro_intermedio = None
    # return False, cuadro
    return cuadro is not None, cuadro


restart(False)
while True:
    # Establecer el cuadro en el que estamos
    num_cuadro += 1

    # Esperar por input del usuario
    key = cv.waitKey(1) & 0xFF
    if key == 27:  # Presionar esc para salir
        break
    elif key == ord(' '):  # Presionar espacio para des/pausar
        running = not running
    elif key == ord('r'):  # Presionar R para reiniciar
        restart(False)
    elif key == ord('p'):
        usar_prediccion = not usar_prediccion

    if not running:  # Si esta pausado,
        continue

    # Leer el cuadro interpolado
    existe, cuadro = tomar_cuadro_intermedio()
    interpolado = existe
    # De lo contrario, leer el cuadro del archivo
    if not existe:
        existe, cuadro = captura.read()

    if not existe:  # Si no hay cuadro, intentar reproducir el cuadro anterior (si existe)
        if cuadro_previo is not None:
            cv.imshow("img", cuadro_previo)
            continue
        else:
            break

    # Procesar el cuadro
    cuadro = imutils.resize(cuadro, width=tamanio_horizontal)  # Redimensionar a 600px de ancho
    difuminado = cv.GaussianBlur(cuadro, (11, 11), 0)  # Difuminar para eliminar ruido
    hsv = cv.cvtColor(difuminado, cv.COLOR_BGR2HSV)

    # Crear la mascara para el color y identificar el objeto
    mask = cv.inRange(hsv, color_lower, color_upper)
    mask = cv.erode(mask, None, iterations=2)
    mask = cv.dilate(mask, None, iterations=2)

    # Obtener la posicion del objeto
    circulos = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # Encontrar contornos
    circulos = imutils.grab_contours(circulos)  # Obtener contornos
    centro = None  # Centro del circulo

    if len(circulos) > 0 and not interpolado:
        # Encontrar el circulo mas grande
        circulo = max(circulos, key=cv.contourArea)
        ((x, y), radioPix) = cv.minEnclosingCircle(circulo)
        M = cv.moments(circulo)
        centro = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # Dibujar circulo y centro
        cv.circle(cuadro, (int(x), int(y)), int(radioPix), (0, 255, 255), 2)
        cv.circle(cuadro, centro, 5, (0, 0, 255), -1)

        if not interpolado and not predicciones.PIXEL_POR_METRO:
            predicciones.PIXEL_POR_METRO = radioPix / radio

    # Predicción si estamos en los 2 puntos iniciales
    if not interpolado and not ha_calculado() and len(puntos_reales) >= 2:
        # Encontrar primeros dos puntos que no sean None
        pos_1, pos_2, cuadros = ultima_posicion()

        # Si no se encontraron dos puntos, no se puede calcular la predicción
        if pos_1 and pos_2:
            angulo_inicial, velocidad_inicial = velocidad_angulo(pos_1[0], pos_1[1], pos_2[0], pos_2[1], cuadros)
            puntos_iniciales.append(pos_1)
            puntos_iniciales.append(pos_2)

    if centro and not posicion_inicial and not interpolado:
        posicion_inicial = centro

    # Pintar predicción si ya se calculo la velocidad inicial y el angulo inicial
    copy = cuadro.copy()
    if ha_calculado():
        # Dibujar angulo y velocidad inicial
        cv.putText(cuadro, f'Angulo Inicial: {round(angulo_inicial, 2)}', (10, 30), cv.FONT_HERSHEY_DUPLEX, 0.65,
                   (0, 0, 0), 2)
        cv.putText(cuadro, f'Velocidad Inicial: {round(velocidad_inicial, 2)} m/s', (10, 60), cv.FONT_HERSHEY_DUPLEX,
                   0.65, (0, 0, 0), 2)

        # Dibujar el angulo graficamente
        x1 = puntos_iniciales[0][0]
        y1 = puntos_iniciales[0][1]
        x2 = puntos_iniciales[1][0]
        y2 = puntos_iniciales[1][1]
        cv.line(cuadro, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.line(cuadro, (x1, y1), (x2, y1), (0, 255, 0), 2)
        cv.circle(cuadro, (x1, y1), 2, (0, 0, 255), -1)
        cv.circle(cuadro, (x2, y2), 2, (0, 0, 255), -1)

        # Dibujar angulo y velocidad actual
        # Encontrar primeros dos puntos que no sean None
        pos_1, pos_2, cuadros = ultima_posicion()

        if pos_1 and pos_2:
            angulo, velocidad = velocidad_angulo(pos_1[0], pos_1[1], pos_2[0], pos_2[1], cuadros)
            cv.putText(cuadro, f'Angulo: {round(angulo, 2)}', (10, 90), cv.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 0), 2)
            cv.putText(cuadro, f'Velocidad: {round(velocidad, 2)} m/s', (10, 120), cv.FONT_HERSHEY_DUPLEX, 0.65,
                       (0, 0, 0), 2)

            # Dibujar predicciones actuales
            if centro and not interpolado:
                # Declarar la cantidad de pixeles a predecir dependiendo de velocidad y angulo
                # actual
                prediccion = int((1 - math.sin(abs(angulo) * pi / 180)) * 95) + 5

                # Prediccion a futuro
                delta_x = pos_2[0] - pos_1[0]
                x = centro[0] + int(math.copysign(prediccion, delta_x))  # Predecir la posición de los siguientes cuadros
                # Tenemos la predicción para los siguientes frames
                _, y = punto(angulo_inicial, velocidad_inicial, x - posicion_inicial[0], metodo)
                y = posicion_inicial[1] - y  # Invertir la y porque la y crece hacia abajo
                puntos_predichos.appendleft((x, y))

                # Prediccion inmediata
                x = centro[0] + int(delta_x / 2)  # Predecir la posición a la mitad del camino entre los dos puntos
                # Tenemos la predicción para el siguiente frame
                _, y = punto(angulo_inicial, velocidad_inicial, x - posicion_inicial[0], metodo)
                y = posicion_inicial[1] - y  # Invertir la y porque la y crece hacia abajo
                posicion_intermedia = (x, y)

                # Además, crear cuadro intermedio
                # Con una prediccion de 1 cuadro, se puede crear un cuadro intermedio
                if len(puntos_predichos) > 0 and radioPix > 0:
                    interpolar(copy)

            # Si hubo un cambio drastico en angulo, hubo un rebote
            # entonces hay que resetear la predicción
            if angulo_previo and abs(angulo - angulo_previo) > 90:
                restart(True)

            angulo_previo = angulo

    # Agregar punto actual a la lista de puntos
    if not interpolado:
        puntos_reales.appendleft(centro)

    # Agregar punto predicho a la lista de puntos
    if usar_prediccion:
        for i in range(1, len(puntos_predichos)):
            # Si alguno de los puntos es None, ignorarlos
            if puntos_predichos[i - 1] is None or puntos_predichos[i] is None:
                continue

            # Si no, dibujar una linea entre el punto anterior y el actual
            grosor = int(buffer / float(i + 1))
            cv.line(cuadro, puntos_predichos[i - 1], puntos_predichos[i], (255, 0, 0), grosor)

    for i in range(1, len(puntos_reales)):
        # Si alguno de los puntos es None, ignorarlos
        if puntos_reales[i - 1] is None or puntos_reales[i] is None:
            continue

        # Si no, dibujar una linea entre el punto anterior y el actual
        grosor = int(buffer / float(i + 1))
        cv.line(cuadro, puntos_reales[i - 1], puntos_reales[i], (0, 0, 255), grosor)

    # Poner tiempo abajo a la izquierda
    transcurrido = time() - start_time
    cv.putText(cuadro, f'Tiempo: {round(transcurrido, 2)}s', (10, cuadro.shape[0] - 10), cv.FONT_HERSHEY_DUPLEX, 0.65,
               (0, 0, 0), 2)

    # Poner tiempo esperado abajo a la derecha
    cv.putText(cuadro, f'Tiempo esperado: {round(tiempo_esperado, 2)}s', (cuadro.shape[1] - 250, cuadro.shape[0] - 10),
               cv.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 0), 2)

    cv.imshow("img", cuadro)  # Mostrar imagen

    # Esperar el tiempo que dura cada cuadro en el video
    # Esto no es necesario si se esta usando la camara web porque el tiempo de espera es el tiempo de procesamiento
    sleep(1 / fps)

    if not interpolado:
        cuadro_previo = cuadro

captura.release()  # Liberar camara
cv.destroyAllWindows()  # Cerrar ventanas
