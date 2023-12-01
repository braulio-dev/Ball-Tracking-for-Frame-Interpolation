# Ball-Tracking-for-Frame-Interpolation

## Iniciar aplicación
```bash
python main.py -a [archivo] -r [radio en metros]
```

Opcionalmente, se puede elegir el metodo numérico a utilizar en el programa de los siguientes: `TAYLOR`, `RUNGE_KUTTA`, `EULER`
```bash
python main.py -a [archivo] -r [radio en metros] -m [método]
```

## Teclas
* Presionar la tecla `R` para reiniciar el video.
* Presionar la tecla `SPACE` para pausar el video.
* Presionar la tecla `ESC` para salir del programa.
* Presionar la tecla `P` para activar o desactivar la cola buffer de predicción de color azul.
