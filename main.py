import argparse
from time import time, sleep

import cv2 as cv
import imutils

import predicciones
from predicciones import *
from collections import deque

args_parser = argparse.ArgumentParser()
args_parser.add_argument("-a", "--archivo", help="Archivo de entrada (opcional)", required=False)
args_parser.add_argument("-r", "--radio", help="Radio estimado, en metros, del objeto a seguir", required=True)
args = vars(args_parser.parse_args())
color_lower = (29, 86, 6)  # Amarillo Verde
color_upper = (64, 255, 255)
# color_lower = (115, 31, 0)  # Rojo
# color_upper = (204, 151, 143)
buffer = 16
tamanio_horizontal = 700
puntos_reales = deque(maxlen=buffer)
puntos_predichos = deque(maxlen=buffer)
radio = float(args["radio"])  # Radio del objeto a seguir

# Tomar archivo de entrada o camara web si no se especifica
if args["archivo"]:
    captura = cv.VideoCapture(args["archivo"])
    predicciones.FPS = fps = captura.get(cv.CAP_PROP_FPS)
    cuadros_totales = captura.get(cv.CAP_PROP_FRAME_COUNT)
    tiempo_esperado = cuadros_totales / fps
else:
    captura = cv.VideoCapture(0)

cuadro_previo = None
running = True
velocidad_inicial = None
angulo_inicial = None
puntos_iniciales = []

num_cuadro = 0
start_time = time()
posicion_inicial = None
while True:
    num_cuadro += 1
    key = cv.waitKey(1) & 0xFF
    if key == 27:  # Presionar esc para salir
        break
    elif key == ord(' '):  # Presionar espacio para des/pausar
        running = not running
    elif args["archivo"] and key == ord('r'):  # Presionar R para reiniciar (solo con archivo de video)
        running = True
        puntos_iniciales.clear()
        puntos_reales.clear()
        puntos_predichos.clear()
        velocidad_inicial = None
        angulo_inicial = None
        num_cuadro = 0
        start_time = time()
        cuadro_previo = None
        captura = cv.VideoCapture(args["archivo"] if args["archivo"] else 0)

    if not running:
        continue

    existe, cuadro = captura.read()
    if not existe:
        if cuadro_previo is not None:
            cv.imshow("img", cuadro_previo)
            continue
        else:
            break

    cuadro = imutils.resize(cuadro, width=tamanio_horizontal)  # Redimensionar a 600px de ancho
    difuminado = cv.GaussianBlur(cuadro, (11, 11), 0)  # Difuminar para eliminar ruido
    hsv = cv.cvtColor(difuminado, cv.COLOR_BGR2HSV)

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv.inRange(hsv, color_lower, color_upper)
    mask = cv.erode(mask, None, iterations=2)
    mask = cv.dilate(mask, None, iterations=2)

    circulos = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # Encontrar contornos
    circulos = imutils.grab_contours(circulos)  # Obtener contornos
    centro = None  # Centro del circulo

    if len(circulos) > 0:
        # Encontrar el circulo mas grande
        circulo = max(circulos, key=cv.contourArea)
        ((x, y), radioPix) = cv.minEnclosingCircle(circulo)
        M = cv.moments(circulo)
        centro = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # Dibujar circulo y centro
        cv.circle(cuadro, (int(x), int(y)), int(radioPix), (0, 255, 255), 2)
        cv.circle(cuadro, centro, 5, (0, 0, 255), -1)

        if not predicciones.PIXEL_POR_METRO:
            predicciones.PIXEL_POR_METRO = radioPix / (4 * radio)
            print(f'PIXEL_POR_METRO: {predicciones.PIXEL_POR_METRO} | radioPix: {radioPix} | radio: {radio}')

    # Predicción si estamos en los 2 puntos iniciales
    if len(puntos_reales) >= 2 and not angulo_inicial and not velocidad_inicial:
        # Encontrar primeros dos puntos que no sean None
        pos_1 = None
        pos_2 = None
        for i in range(0, len(puntos_reales)):
            if puntos_reales[i] is not None:
                pos_2 = puntos_reales[i]
                break
        for i in range(i + 1, len(puntos_reales)):
            if puntos_reales[i] is not None:
                pos_1 = puntos_reales[i]
                break

        # Si no se encontraron dos puntos, no se puede calcular la predicción
        if pos_1 and pos_2:
            angulo_inicial, velocidad_inicial = VelocidadAngulo(pos_1[0], pos_1[1], pos_2[0], pos_2[1], 2)
            puntos_iniciales.append(pos_1)
            puntos_iniciales.append(pos_2)

    if centro and not posicion_inicial:
        posicion_inicial = centro

    # Pintar predicción si ya se calculo la velocidad inicial y el angulo inicial
    if velocidad_inicial and angulo_inicial:
        # Dibujar angulo y velocidad inicial
        cv.putText(cuadro, f'Angulo Inicial: {round(angulo_inicial, 2)}', (10, 30), cv.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 0), 2)
        cv.putText(cuadro, f'Velocidad Inicial: {round(velocidad_inicial, 2)} m/s', (10, 60), cv.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 0), 2)

        # Dibujar angulo y velocidad actual
        # Encontrar primeros dos puntos que no sean None
        pos_1 = None
        pos_2 = None
        for i in range(0, len(puntos_reales)):
            if puntos_reales[i] is not None:
                pos_2 = puntos_reales[i]
                break
        for i in range(i + 1, len(puntos_reales)):
            if puntos_reales[i] is not None:
                pos_1 = puntos_reales[i]
                break

        if pos_1 and pos_2:
            angulo, velocidad = VelocidadAngulo(pos_1[0], pos_1[1], pos_2[0], pos_2[1], 2)
            cv.putText(cuadro, f'Angulo: {round(angulo, 2)}', (10, 90), cv.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 0), 2)
            cv.putText(cuadro, f'Velocidad: {round(velocidad, 2)} m/s', (10, 120), cv.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 0), 2)

        # Dibujar el angulo graficamente
        x1 = puntos_iniciales[0][0]
        y1 = puntos_iniciales[0][1]
        x2 = puntos_iniciales[1][0]
        y2 = puntos_iniciales[1][1]
        cv.line(cuadro, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.line(cuadro, (x1, y1), (x2, y1), (0, 255, 0), 2)
        cv.circle(cuadro, (x1, y1), 2, (0, 0, 255), -1)
        cv.circle(cuadro, (x2, y2), 2, (0, 0, 255), -1)

        # Dibujar predicciones
        if centro and args["archivo"]:
            x = centro[0] + 100  # Predecir la posición de los siguientes cuadros
            # Tenemos la predicción para los siguientes frames
            x, y = Punto(angulo_inicial, velocidad_inicial, x)
            y = cuadro.shape[0] - y - posicion_inicial[1]  # Invertir la y porque la y crece hacia abajo
            puntos_predichos.appendleft((x, y))

    puntos_reales.appendleft(centro)
    for i in range(1, len(puntos_reales)):
        # Si alguno de los puntos es None, ignorarlos
        if puntos_reales[i - 1] is None or puntos_reales[i] is None:
            continue

        # Si no, dibujar una linea entre el punto anterior y el actual
        grosor = int(buffer / float(i + 1))
        cv.line(cuadro, puntos_reales[i - 1], puntos_reales[i], (0, 0, 255), grosor)

    for i in range(1, len(puntos_predichos)):
        # Si alguno de los puntos es None, ignorarlos
        if puntos_reales[i - 1] is None or puntos_reales[i] is None:
            continue

        # Si no, dibujar una linea entre el punto anterior y el actual
        grosor = int(buffer / float(i + 1))
        cv.line(cuadro, puntos_predichos[i - 1], puntos_predichos[i], (255, 0, 0), grosor)

    # Poner tiempo abajo a la izquierda
    transcurrido = time() - start_time
    cv.putText(cuadro, f'Tiempo: {round(transcurrido, 2)}s', (10, cuadro.shape[0] - 10), cv.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 0), 2)

    # Poner tiempo esperado abajo a la derecha
    if (args["archivo"]):
        cv.putText(cuadro, f'Tiempo esperado: {round(tiempo_esperado, 2)}s', (cuadro.shape[1] - 250, cuadro.shape[0] - 10), cv.FONT_HERSHEY_DUPLEX, 0.65, (0, 0, 0), 2)

    cv.imshow("img", cuadro)  # Mostrar imagen

    # Hacer la mascara del tamaño del cuadro
    if args["archivo"]:
        # Esperar el tiempo que dura cada cuadro en el video
        # Esto no es necesario si se esta usando la camara web porque el tiempo de espera es el tiempo de procesamiento
        sleep(1 / fps)

    cuadro_previo = cuadro

captura.release()  # Liberar camara
cv.destroyAllWindows()  # Cerrar ventanas
