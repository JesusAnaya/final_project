# -----------------------------------------------------------------------------
# El código dentro de este módulo, corresponde al controlador que desarrollamos 
# como equipo para recolectar datos de entrenamiento para el modelo de IA.
# Este controlador se encarga de:
# - Obtener la imagen de la cámara
# - Preprocesar la imagen para el modelo de IA
# - Guardar las imágenes en un directorio
# - Guardar los datos en un archivo CSV
# -----------------------------------------------------------------------------


from controller import Display, Keyboard, Robot, Camera, Supervisor
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os
import time
import csv


# Directorio base para almacenar los datasets de entrenamiento
BASE_DIR = "./datasets"


# Función para obtener la imagen de la cámara y convertirla al formato correcto
def get_image(camera):
    # Obtenemos la imagen raw de la cámara
    raw_image = camera.getImage()  
    # Convertimos el buffer de bytes a un array numpy con forma (altura, ancho, 4)
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    # Cambiamos a formato RGB eliminando el canal alpha
    image = image[:, :, [2, 1, 0]]
    return image

# Función para extraer la región de interés (ROI) de la imagen
def get_roi_image(image):
    # Obtenemos las dimensiones de la imagen y recortamos la mitad inferior
    h, w, _ = image.shape
    cropped = image[h//2:h, 0:w]  # Tomamos la mitad inferior (80 filas)
    
    # Redimensionamos a las dimensiones objetivo usando interpolación por área
    resized_img = cv2.resize(cropped, (200, 66), interpolation=cv2.INTER_AREA)

    return resized_img

# Función para mostrar la imagen en el display del simulador
def display_image(display, image):
    roi_img = get_roi_image(image)
    
    # Aseguramos que el array tenga un layout contiguo en memoria y lo convertimos a bytes
    rgb_contiguous = np.ascontiguousarray(roi_img)
    data = rgb_contiguous.tobytes()
    
    # Creamos y mostramos la imagen en el display
    image_ref = display.imageNew(
        data,
        Display.RGB,
        width=200,
        height=66
    )
    display.imagePaste(image_ref, 0, 0, False)

# Variables globales para el control del vehículo
angle = 0.0  # Ángulo inicial de dirección
speed = 15   # Velocidad inicial en km/h
is_recording = False  # Bandera para controlar el estado de grabación
recording_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Nombre del archivo de grabación
last_recording_time = time.time()  # Tiempo de la última grabación
space_pressed = False  # Estado de la tecla espacio
up_down_pressed = False  # Estado de las teclas arriba/abajo

# Función para ajustar la velocidad del vehículo
def set_speed(offset_speed):
    global speed
    
    # Limitamos la velocidad entre 0 y 40 km/h
    speed = np.clip(speed + offset_speed, 0, 40)

    print(f"Velocidad ajustada a {speed} km/h")

# Función para establecer el ángulo de dirección con límites de seguridad
def set_steering_angle(wheel_angle):
    global angle, steering_angle
    # Verificamos límites de cambio de dirección
    if (wheel_angle - steering_angle) > 0.1:
        wheel_angle = steering_angle + 0.1
    if (wheel_angle - steering_angle) < -0.1:
        wheel_angle = steering_angle - 0.1
    steering_angle = wheel_angle
  
    # Limitamos el rango del ángulo de dirección
    if wheel_angle > 0.5:
        wheel_angle = 0.5
    elif wheel_angle < -0.5:
        wheel_angle = -0.5
    # Actualizamos el ángulo de dirección
    angle = wheel_angle

# Función para validar y aplicar incrementos al ángulo de dirección
def change_steer_angle(inc):
    global manual_steering
    # Aplicamos el incremento
    new_manual_steering = manual_steering + inc
    # Validamos que el ángulo esté dentro del rango permitido
    if new_manual_steering <= 25.0 and new_manual_steering >= -25.0: 
        manual_steering = new_manual_steering
        # Redondeamos para asegurar múltiplos exactos de 0.01
        set_steering_angle(round(manual_steering * 0.01, 4))
    # Información de depuración
    if manual_steering == 0:
        print("Dirección centrada")
    else:
        turn = "izquierda" if steering_angle < 0 else "derecha"
        print(f"Girando {str(steering_angle)} rad hacia la {turn}")
        print(f"Dirección manual: {manual_steering}")

# Constantes para el control de la dirección
STEERING_INCREMENT = 0.02  # Incremento fijo en radianes por pulsación de tecla
MAX_STEERING_ANGLE = 0.5  # Ángulo máximo de dirección en radianes
STEERING_SMOOTHING = 0.1  # Cambio máximo permitido por paso de tiempo

# Estado de la dirección
target_steering = 0.0  # Ángulo objetivo de la dirección
current_steering = 0.0  # Ángulo actual suavizado de la dirección
prev_keys = {
    'LEFT': False,  # Estado anterior de la tecla izquierda
    'RIGHT': False  # Estado anterior de la tecla derecha
}

def update_steering():
    """Actualiza el ángulo de dirección aplicando un suavizado para movimientos más naturales.
    
    Esta función:
    - Calcula la diferencia entre el ángulo objetivo y el actual
    - Aplica un suavizado para evitar cambios bruscos
    - Limita la velocidad de cambio del ángulo
    """
    global current_steering, target_steering
    # Interpolamos suavemente la dirección actual hacia el objetivo
    diff = target_steering - current_steering
    if abs(diff) > STEERING_SMOOTHING:
        if diff > 0:
            current_steering += STEERING_SMOOTHING
        else:
            current_steering -= STEERING_SMOOTHING
    else:
        current_steering = target_steering
    return current_steering

def handle_steering_keys(key, keyboard):
    """Procesa las entradas del teclado para el control de la dirección.
    
    Esta función:
    - Detecta las pulsaciones de las teclas izquierda/derecha
    - Actualiza el ángulo objetivo de dirección
    - Aplica límites de seguridad al ángulo
    - Mantiene un registro del estado anterior de las teclas
    """
    global target_steering, prev_keys
    
    # Detectamos eventos de pulsación de teclas
    left_pressed = key == keyboard.LEFT and not prev_keys['LEFT']
    right_pressed = key == keyboard.RIGHT and not prev_keys['RIGHT']
    
    # Actualizamos estados de las teclas
    prev_keys['LEFT'] = key == keyboard.LEFT
    prev_keys['RIGHT'] = key == keyboard.RIGHT
    
    # Aplicamos cambios de dirección solo en la pulsación inicial
    if left_pressed:
        target_steering = max(target_steering - STEERING_INCREMENT, -MAX_STEERING_ANGLE)
        print(f"Girando a la izquierda: {target_steering:.2f} rad")
    elif right_pressed:
        target_steering = min(target_steering + STEERING_INCREMENT, MAX_STEERING_ANGLE)
        print(f"Girando a la derecha: {target_steering:.2f} rad")


def start_recording():
    """Inicia una nueva sesión de grabación de datos.
    
    Esta función:
    - Crea un nuevo directorio con marca de tiempo
    - Inicializa el archivo CSV para almacenar los datos
    - Configura las columnas del archivo CSV
    """
    global is_recording, recording_name

    if not is_recording:
        return
    
    # Generamos un nombre único con marca de tiempo
    recording_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    # Creamos una carpeta para almacenar las imágenes
    folder_name = f"{BASE_DIR}/{recording_name}"
    os.makedirs(folder_name, exist_ok=True)
    
    # Creamos y configuramos el archivo CSV para los datos
    csv_file_name = f"{BASE_DIR}/{recording_name}.csv"
    csv_file = open(csv_file_name, "w")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["imagen_izquierda", "imagen_central", "imagen_derecha", "angulo_direccion", "marca_tiempo"])
    csv_file.close()


def perform_recording(center_camera, left_camera, right_camera, angle):
    """Graba los datos de las tres cámaras y el ángulo de dirección.
    
    Esta función:
    - Captura imágenes de las tres cámaras cada 200ms
    - Procesa y guarda las imágenes en formato BGR
    - Registra los datos en el archivo CSV
    - Mantiene un control del tiempo entre grabaciones
    """
    global is_recording, recording_name, last_recording_time
    if not is_recording:
        return
    
    # Solo grabamos cada 200ms para evitar datos redundantes
    if time.time() - last_recording_time < 0.2:
        return
    
    last_recording_time = time.time()
    
    # Obtenemos la marca de tiempo actual
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

    # Capturamos imágenes de las tres cámaras
    image = get_image(center_camera)
    image_left = get_image(left_camera)
    image_right = get_image(right_camera)

    # Extraemos la región de interés de cada imagen
    roi_image_center = get_roi_image(image)
    roi_image_left = get_roi_image(image_left)
    roi_image_right = get_roi_image(image_right)

    # Convertimos de RGB a BGR para guardar con OpenCV
    roi_image_center = cv2.cvtColor(roi_image_center, cv2.COLOR_RGB2BGR)
    roi_image_left = cv2.cvtColor(roi_image_left, cv2.COLOR_RGB2BGR)
    roi_image_right = cv2.cvtColor(roi_image_right, cv2.COLOR_RGB2BGR)

    # Definimos las rutas de las imágenes
    image_center_path = f"{recording_name}/{timestamp}_imagen_central.jpg"
    image_left_path = f"{recording_name}/{timestamp}_imagen_izquierda.jpg"
    image_right_path = f"{recording_name}/{timestamp}_imagen_derecha.jpg"
    
    # Guardamos las imágenes procesadas
    cv2.imwrite(BASE_DIR + "/" + image_center_path, roi_image_center)
    cv2.imwrite(BASE_DIR + "/" + image_left_path, roi_image_left)
    cv2.imwrite(BASE_DIR + "/" + image_right_path, roi_image_right)

    # Registramos los datos en el archivo CSV
    csv_file = open(f"{BASE_DIR}/{recording_name}.csv", "a")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([image_center_path, image_left_path, image_right_path, f"{angle:.4f}", timestamp])
    csv_file.close()

    print(f"Grabación realizada en {timestamp}")


# Función principal del programa
def main():
    """Función principal que controla el vehículo y la grabación de datos.
    
    Esta función:
    - Inicializa el robot y sus componentes
    - Configura las tres cámaras (central, izquierda y derecha)
    - Maneja la interacción del usuario mediante el teclado
    - Controla la grabación de datos para el entrenamiento
    - Actualiza la visualización en tiempo real
    """
    global recording_name, up_down_pressed, space_pressed, is_recording

    # Creamos la instancia del Robot
    robot = Car()
    driver = Driver()
    sup = Supervisor()

    # Obtenemos el paso de tiempo del mundo de simulación
    timestep = int(robot.getBasicTimeStep())

    # Configuración de las cámaras
    # Nota: Las cámaras se configuran con los siguientes parámetros:
    # - Resolución: 640x480 píxeles
    # - Campo de visión: 90 grados
    # - Posición: Centrada en el vehículo
    # - Orientación: Mirando hacia adelante
    camera = robot.getDevice("camera_center")
    camera.enable(timestep)

    # Configuramos las cámaras laterales
    camera_left = robot.getDevice("camera_left")
    camera_left.enable(timestep)
    camera_right = robot.getDevice("camera_right")
    camera_right.enable(timestep)

    # Configuramos el display para procesamiento de imágenes
    display_img = Display("display")

    # Creamos y habilitamos la instancia del teclado
    keyboard = Keyboard()
    keyboard.enable(timestep)

    # Bucle principal del programa
    while robot.step() != -1:
        # Obtenemos la imagen de la cámara
        image = get_image(camera)

        # Procesamos y mostramos la imagen
        display_image(display_img, image)

        # Leemos las entradas del teclado
        key = keyboard.getKey()

        # Manejamos el control de velocidad
        if key == keyboard.UP:
            if not up_down_pressed:
                set_speed(5.0)
                print("Aumentando velocidad")
            up_down_pressed = True 

        elif key == keyboard.DOWN:
            if not up_down_pressed:
                set_speed(-5.0)
                print("Reduciendo velocidad")
            up_down_pressed = True

        elif key == ord(' '):
            if not space_pressed:  # Solo cambiamos si la tecla espacio no estaba presionada
                is_recording = not is_recording

                # Iniciamos la grabación si está activada
                start_recording()

                print(f"Grabación: {'ACTIVADA' if is_recording else 'DESACTIVADA'}")
            space_pressed = True
            
        elif space_pressed and key == -1:  # Tecla espacio liberada
            space_pressed = False

        elif up_down_pressed and key == -1:
            up_down_pressed = False

        else:
            # Manejamos las teclas de dirección
            handle_steering_keys(key, keyboard)
            
        # Actualizamos el ángulo de giro del vehículo
        angle = update_steering()
            
        # Aplicamos el ángulo y la velocidad al vehículo
        driver.setSteeringAngle(angle)
        driver.setCruisingSpeed(speed)

        # Realizamos la grabación de los datos, si está activada la bandera is_recording.
        perform_recording(camera, camera_left, camera_right, angle)

        # Mostramos información en la pantalla del supervisor
        sup.setLabel(
            0,
            f"Ángulo de dirección: {angle:.2f}",
            0.05,    # x = 5% desde la izquierda
            0.85,    # y = 85% desde abajo
            0.05,    # altura del texto = 5% de la pantalla
            0xFFFFFF,# color blanco
            0.0,     # completamente opaco
            "Arial"  # nombre de la fuente
        )

        sup.setLabel(
            1,
            f"Velocidad: {speed} km/h",
            0.05,    # x = 5% desde la izquierda
            0.90,    # y = 90% desde abajo
            0.05,    # altura del texto = 5% de la pantalla
            0xFFFFFF,# color blanco
            0.0,     # completamente opaco
            "Arial"  # nombre de la fuente
        )

        # Mostramos el estado de la grabación
        sup.setLabel(
            2,
            f"Grabación: {'ACTIVADA' if is_recording else 'DESACTIVADA'}",
            0.05,    # x = 5% desde la izquierda
            0.95,    # y = 95% desde abajo
            0.05,    # altura del texto = 5% de la pantalla
            0xFFFFFF,# color blanco
            0.0,     # completamente opaco
            "Arial"  # nombre de la fuente
        )


if __name__ == "__main__":
    main()