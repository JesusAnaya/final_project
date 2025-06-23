# -----------------------------------------------------------------------------
# El código dentro de este módulo, corresponde al controlador que desarrollamos 
# como equipo para controlar el vehículo en el simulador.
# Este controlador se encarga de:
# - Obtener la imagen de la cámara
# - Procesar la imagen para el modelo de IA
# - Predecir el ángulo de dirección
# - Aplicar el ángulo predicho con límites de seguridad
# - Controlar la velocidad del vehículo
# - Controlar la dirección del vehículo
# -----------------------------------------------------------------------------

from controller import Display, Keyboard, Robot, Camera, Supervisor
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import layers
import time
import numpy as np

# Ruta del modelo pre-entrenado de NVIDIA para la conducción autónoma
MODEL_PATH_NVIDIA = "./models/best_model_cpu_v8.h5"
# Cargamos el modelo de keras sin compilar para hacer predicciones
keras_model = load_model(MODEL_PATH_NVIDIA, compile=False)


# Función para obtener la imagen de la cámara y convertirla al formato correcto
def get_image(camera):
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
    # Extraemos solo la parte inferior de la imagen que contiene la carretera
    h, w, _ = image.shape
    cropped = image[h//2:h, 0:w]  # Extraemos solo la parte de abajo de la imagen
    
    # Redimensionamos la imagen a las dimensiones que el modelo espera (200x66)
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
is_auto_driving = False  # Estado de conducción autónoma
space_pressed = False    # Estado de la tecla espacio
up_down_pressed = False  # Estado de las teclas arriba/abajo
steering_angle = 0.0     # Ángulo actual de dirección

# Variables para el control de tiempo de predicción
last_prediction_time = 0
PREDICTION_INTERVAL = 0.2  # Intervalo de 200ms entre predicciones

# Constantes para la detección de objetos
RADAR_MIN_RANGE = 2.5  # Rango mínimo del radar en metros
RADAR_MAX_RANGE = 30.0  # Rango máximo del radar en metros
EMERGENCY_RANGE = 8.0  # Distancia de frenado de emergencia en metros
MID_RANGE = 12.0  # Distancia para reducción de velocidad en metros

# Configuración de los sensores
FRONT_ANGLE_THRESHOLD = np.pi/2  # 90 grados - ángulo amplio para mejor detección en curvas

# Configuración de velocidades
EMERGENCY_SPEED = 0.0  # km/h (parada completa)
MEDIUM_SPEED = 5.0  # km/h (velocidad reducida cuando hay objetos a media distancia)
NORMAL_SPEED = 15.0  # km/h (velocidad normal de crucero)
MAX_SPEED = 40.0  # km/h (velocidad máxima permitida)

# Configuración de tiempos de frenado
BRAKE_RELEASE_TIME = 3.0  # Segundos para mantener el freno después de que el obstáculo se despeje
SPEED_CHANGE_TIME = 1.5  # Segundos para mantener la velocidad reducida después de la advertencia
last_brake_time = 0  # Tiempo del último frenado
last_warning_time = 0  # Tiempo de la última advertencia
is_braking = False  # Indica si el vehículo está frenando
is_warning = False  # Indica si el vehículo está en estado de advertencia

# Variables de monitoreo para el radar
monitor_radar_min_dist = 0.0  # Distancia mínima detectada por cualquier radar
monitor_radar_position = "none"  # Posición del radar que detectó el objeto más cercano

# Parámetros de control del vehículo
EMERGENCY_BRAKE_INTENSITY = 1.0  # Intensidad máxima de frenado
MEDIUM_BRAKE_INTENSITY = 0.4  # Intensidad media de frenado
MEDIUM_THROTTLE = 0.3  # Aceleración media
NORMAL_THROTTLE = 1.0  # Aceleración normal

# Constantes para el control de la dirección
STEERING_INCREMENT = 0.02  # Incremento fijo en radianes por pulsación de tecla
MAX_STEERING_ANGLE = 0.5  # Ángulo máximo de dirección en radianes
STEERING_SMOOTHING = 0.1  # Cambio máximo permitido por paso de tiempo

# Estado de la dirección
target_steering = 0.0  # Ángulo objetivo de la dirección
current_steering = 0.0  # Ángulo actual suavizado de la dirección
prev_keys = {
    'LEFT': False,
    'RIGHT': False
}


def process_radar_data(radar):
    """Procesa los datos de un radar y retorna la distancia mínima válida detectada.
    
    Esta función filtra las lecturas del radar para:
    - Eliminar falsos positivos (lecturas exactamente a 1.00m)
    - Considerar solo objetos dentro del cono frontal
    - Validar que las distancias estén dentro del rango operativo del radar
    """
    radar_dists = []
    radar_targets = radar.getTargets()
    
    for target in radar_targets:
        # Ignoramos lecturas exactamente a 1.00m por ser probablemente falsos positivos
        if abs(target.distance - 1.00) < 0.01:
            continue
            
        horizontal_angle = target.azimuth
            
        # Solo consideramos objetivos dentro del cono frontal y rango válido
        if (abs(horizontal_angle) <= FRONT_ANGLE_THRESHOLD and
            RADAR_MIN_RANGE <= target.distance <= RADAR_MAX_RANGE):
            radar_dists.append(target.distance)
    
    return min(radar_dists) if radar_dists else float('inf')


# Ya que en este vehículo contamos con 3 radares, se debe combinar la información de 
# los 3 radares para obtener una mejor detección de obstáculos.
# Para ello, se debe obtener la distancia mínima de cada radar y 
# luego compararla con los límites de distancia para frenado de emergencia, 
# reducción de velocidad y velocidad normal.
# Además, se debe mantener un estado de frenado o advertencia durante un tiempo mínimo.
def combined_radar_control(radar_center, radar_left, radar_right, current_speed, driver):
    """Función principal de control basado en los tres radares.
    
    Analiza las lecturas de los tres radares (central, izquierdo y derecho) para:
    - Detectar obstáculos y su proximidad
    - Determinar acciones de frenado o reducción de velocidad
    - Mantener un estado de frenado o advertencia durante un tiempo mínimo
    """
    global monitor_radar_min_dist, monitor_radar_position, last_brake_time, last_warning_time, is_braking, is_warning
    current_time = time.time()
    
    # Obtenemos la distancia mínima de cada radar
    center_dist = process_radar_data(radar_center)
    left_dist = process_radar_data(radar_left)
    right_dist = process_radar_data(radar_right)
    
    # Encontramos la detección más cercana y qué radar la detectó
    radar_distances = {
        "center": center_dist,
        "left": left_dist,
        "right": right_dist
    }
    
    min_dist = min(radar_distances.values())
    monitor_radar_min_dist = min_dist
    monitor_radar_position = min(radar_distances.items(), key=lambda x: x[1])[0]

    # Verificamos si necesitamos iniciar el frenado de emergencia
    if min_dist <= EMERGENCY_RANGE:
        print(f"🚨 Frenado de emergencia - objeto detectado a {min_dist:.1f}m por el radar {monitor_radar_position}")
        driver.setThrottle(0.0)
        driver.setBrakeIntensity(EMERGENCY_BRAKE_INTENSITY)
        last_brake_time = current_time
        is_braking = True
        is_warning = False  # Reseteamos estado de advertencia en emergencia
        return EMERGENCY_SPEED
    
    # Verificamos si necesitamos reducir la velocidad
    elif EMERGENCY_RANGE < min_dist <= MID_RANGE:
        print(f"⚠️ Reduciendo velocidad - objeto detectado a {min_dist:.1f}m por el radar {monitor_radar_position}")
        driver.setBrakeIntensity(MEDIUM_BRAKE_INTENSITY)
        driver.setThrottle(MEDIUM_THROTTLE)
        last_warning_time = current_time
        is_warning = True
        is_braking = False  # Reseteamos estado de emergencia en advertencia
        return MEDIUM_SPEED
    
    # Verificamos los temporizadores para ambos estados
    if is_braking and (current_time - last_brake_time) < BRAKE_RELEASE_TIME:
        # Mantenemos estado de frenado de emergencia
        return current_speed
    elif is_warning and (current_time - last_warning_time) < SPEED_CHANGE_TIME:
        # Mantenemos estado de velocidad de advertencia
        return MEDIUM_SPEED
    
    # Si llegamos aquí, no hay obstáculos o los temporizadores expiraron
    is_braking = False
    is_warning = False
    driver.setBrakeIntensity(0.0)
    driver.setThrottle(NORMAL_THROTTLE)
    return NORMAL_SPEED


def set_speed(offset_speed):
    """Ajusta la velocidad del vehículo dentro de los límites permitidos."""
    global speed
    
    speed = np.clip(speed + offset_speed, 0, 40)
    print(f"Velocidad ajustada a {speed} km/h")


def set_steering_angle(wheel_angle):
    """Establece el ángulo de dirección con límites de seguridad y suavizado.
    
    Aplica restricciones para:
    - Limitar el cambio máximo por paso de tiempo
    - Mantener el ángulo dentro de los límites seguros
    """
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


def change_steer_angle(inc):
    """Cambia el ángulo de dirección manual con incrementos controlados."""
    global manual_steering
    # Aplicamos el incremento
    new_manual_steering = manual_steering + inc
    # Validamos el intervalo permitido
    if new_manual_steering <= 25.0 and new_manual_steering >= -25.0: 
        manual_steering = new_manual_steering
        # Redondeamos para asegurar múltiplos exactos de 0.02
        set_steering_angle(round(manual_steering * 0.02, 2))
    # Información de depuración
    if manual_steering == 0:
        print("Dirección centrada")
    else:
        turn = "izquierda" if steering_angle < 0 else "derecha"
        print(f"Girando {str(steering_angle)} rad hacia la {turn}")
        print(f"Dirección manual: {manual_steering}")


def update_steering():
    """Actualiza el ángulo de dirección con suavizado para movimientos más naturales."""
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


steering_angle_median_buffer = []

def steering_angle_median(angle, buffer_size=10):
    """Calcula la mediana del ángulo de dirección para suavizar movimientos bruscos."""
    # Mantenemos solo los últimos 10 valores
    steering_angle_median_buffer.append(angle)
    if len(steering_angle_median_buffer) > buffer_size:
        steering_angle_median_buffer.pop(0)
    return np.median(steering_angle_median_buffer)


def perform_auto_driving(camera):
    """Realiza la conducción autónoma utilizando el modelo de IA.
    
    Esta función:
    - Captura imágenes de la cámara
    - Preprocesa las imágenes para el modelo
    - Predice el ángulo de dirección
    - Aplica el ángulo predicho con límites de seguridad
    """
    global is_auto_driving, target_steering, last_prediction_time

    if not is_auto_driving:
        return
    
    # Verificamos si ha pasado suficiente tiempo desde la última predicción
    current_time = time.time()
    if current_time - last_prediction_time < PREDICTION_INTERVAL:
        return
        
    # Obtenemos la imagen de la cámara
    image = get_image(camera)
    
    # Obtenemos la región de interés
    roi_image = get_roi_image(image)              # NumPy uint8 en [0,255]
    roi_image = roi_image.astype('float32') / 255.0  # Normalizamos a [0,1]

    # Convertimos la imagen a tensor
    roi_image = tf.convert_to_tensor(roi_image, dtype=tf.float32)
    roi_image = tf.expand_dims(roi_image, axis=0)  # Agregamos dimensión de batch
    
    # Realizamos la predicción del ángulo
    angle_pred = keras_model.predict(roi_image, verbose=0)
    angle = float(angle_pred[0][0])

    print(f"Ángulo predicho: {angle:.9f} rad")
    
    # Limitamos el ángulo predicho a los valores seguros
    target_steering = np.clip(angle, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)
    
    # Actualizamos el tiempo de la última predicción
    last_prediction_time = current_time


def handle_steering_keys(key, keyboard):
    """Maneja las teclas de dirección para el control manual."""
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


# main
def main():
    """Función principal del programa de conducción autónoma.
    
    Esta función:
    - Inicializa el robot y sus componentes
    - Configura los sensores y la cámara
    - Ejecuta el bucle principal de control
    - Maneja la interacción del usuario y la visualización
    """
    global recording_name, up_down_pressed, space_pressed, is_auto_driving

    # Verificamos la compatibilidad de CUDA con TensorFlow
    print(tf.config.list_physical_devices('GPU'))
    speed = 15  # Velocidad inicial del vehículo

    # Creamos la instancia del Robot
    robot = Car()
    print("Robot conectado")

    driver = Driver()
    sup = Supervisor()

    # Obtenemos el paso de tiempo del mundo actual
    timestep = int(robot.getBasicTimeStep())
    print("Paso de tiempo obtenido")

    # Creamos la instancia de la cámara y la habilitamos
    camera = robot.getDevice("camera")
    print("Cámara obtenida")
    camera.enable(timestep)

    # Obtenemos y habilitamos los dispositivos de radar
    radar = robot.getDevice("front_radar")
    radar.enable(timestep)

    radar_left = robot.getDevice("front_radar_left")
    radar_left.enable(timestep)

    radar_right = robot.getDevice("front_radar_right")
    radar_right.enable(timestep)

    # Configuramos el display para procesamiento
    display_img = Display("display")
    print("Display procesado")

    # Creamos y habilitamos la instancia del teclado
    keyboard = Keyboard()
    keyboard.enable(timestep)
    print("Teclado habilitado")

    # Bucle principal del programa
    while robot.step() != -1:
        # Obtenemos la imagen de la cámara
        image = get_image(camera)

        # Procesamos y mostramos la imagen
        display_image(display_img, image)

        # Leemos el teclado
        key = keyboard.getKey()

        # Manejamos las teclas de control de velocidad
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
            if not space_pressed:  # Solo cambiamos si la tecla espacio no estaba presionada antes
                is_auto_driving = not is_auto_driving
                print(f"Conducción Autónoma: {'ACTIVADA' if is_auto_driving else 'DESACTIVADA'}")
            space_pressed = True
            
        elif space_pressed and key == -1:  # Tecla espacio liberada
            space_pressed = False

        elif up_down_pressed and key == -1:
            up_down_pressed = False

        else:
            # Manejamos las teclas de dirección
            handle_steering_keys(key, keyboard)
            
        # Actualizamos el ángulo de la dirección del vehículo
        angle = update_steering()
            
        # Actualizamos el ángulo de la dirección del vehículo
        driver.setSteeringAngle(angle)

        # Obtenemos la velocidad basada en las lecturas de los sensores
        speed = combined_radar_control(radar, radar_left, radar_right, speed, driver)

        # Actualizamos la velocidad del vehículo
        driver.setCruisingSpeed(speed)

        # Realizamos la predicción de la dirección del vehículo
        perform_auto_driving(camera)

        # Mostramos información en la pantalla del supervisor
        sup.setLabel(
            0,
            f"Objeto más cercano: {monitor_radar_min_dist:.2f}m (radar {monitor_radar_position})",
            0.05,    # x = 5% desde la izquierda
            0.80,    # y = 80% desde abajo
            0.05,    # altura del texto = 5% de la pantalla
            0xFFFFFF,# color blanco
            0.0,     # completamente opaco
            "Arial"  # nombre de la fuente
        )

        sup.setLabel(
            1,
            f"Ángulo de dirección: {angle:.2f}",
            0.05,    # x = 5% desde la izquierda
            0.85,    # y = 85% desde abajo
            0.05,    # altura del texto = 5% de la pantalla
            0xFFFFFF,# color blanco
            0.0,     # completamente opaco
            "Arial"  # nombre de la fuente
        )

        sup.setLabel(
            2,
            f"Velocidad: {speed} km/h",
            0.05,    # x = 5% desde la izquierda
            0.90,    # y = 90% desde abajo
            0.05,    # altura del texto = 5% de la pantalla
            0xFFFFFF,# color blanco
            0.0,     # completamente opaco
            "Arial"  # nombre de la fuente
        )

        # Mostramos el estado de la conducción autónoma
        sup.setLabel(
            3,
            f"Conducción Autónoma: {'ACTIVADA' if is_auto_driving else 'DESACTIVADA'}",
            0.05,    # x = 5% desde la izquierda
            0.95,    # y = 95% desde abajo
            0.05,    # altura del texto = 5% de la pantalla
            0xFFFFFF,# color blanco
            0.0,     # completamente opaco
            "Arial"  # nombre de la fuente
        )


if __name__ == "__main__":
    main()