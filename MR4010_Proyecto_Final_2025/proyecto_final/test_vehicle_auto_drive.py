from controller import Display, Keyboard, Robot, Camera, Supervisor
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import layers
import os
import time
import csv
import numpy as np


MODEL_PATH_NVIDIA = "./models/best_model_cpu_v8.h5"
keras_model = load_model(MODEL_PATH_NVIDIA, compile=False)


# Capa para redimensionar y rellenar imágenes
class PadAndResize(layers.Layer):
    def __init__(self, pad_to=200, resize_to=224, **kwargs):
        super().__init__(**kwargs)
        self.pad_to = pad_to
        self.resize_to = resize_to

    def call(self, inputs):
        # inputs: tensor de forma (batch, 66, 200, 3)
        shape = tf.shape(inputs)
        height = shape[1]
        # calcular cuántos píxeles agregar arriba
        pad_top = self.pad_to - height  # 200 - 66 = 134
        # pad de 134 arriba, 0 abajo, 0/0 en ancho, 0 en canales
        padded = tf.image.pad_to_bounding_box(
            inputs,
            offset_height=pad_top,
            offset_width=0,
            target_height=self.pad_to,
            target_width=self.pad_to
        )
        # redimensionar de 200×200 a resize_to×resize_to
        resized = tf.image.resize(padded, [self.resize_to, self.resize_to])
        return resized

    def get_config(self):
        config = super().get_config()
        config.update({
            "pad_to": self.pad_to,
            "resize_to": self.resize_to
        })
        return config


#Getting image from camera
def get_image(camera):
    raw_image = camera.getImage()  
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    # change to RGB
    image = image[:, :, [2, 1, 0]]
    return image

def get_roi_image(image):
    # Extraemos solo la parte de la imagen que nos interesa
    h, w, _ = image.shape
    cropped = image[h//2:h, 0:w]  # Extraemos solo la parte de abajo de la imagen
    
    # Redimensionamos la imagen a las dimensiones que el modelo espera
    resized_img = cv2.resize(cropped, (200, 66), interpolation=cv2.INTER_AREA)

    return resized_img

#Display image 
def display_image(display, image):
    roi_img = get_roi_image(image)
    
    # Ensure C-contiguous layout and convert to bytes
    rgb_contiguous = np.ascontiguousarray(roi_img)
    data = rgb_contiguous.tobytes()
    
    # Create and display image
    image_ref = display.imageNew(
        data,
        Display.RGB,
        width=200,
        height=66
    )
    display.imagePaste(image_ref, 0, 0, False)

#initial angle and speed 
angle = 0.0
speed = 15
is_auto_driving = False
space_pressed = False
up_down_pressed = False
steering_angle = 0.0

# Add this at the top with other global variables
last_prediction_time = 0
PREDICTION_INTERVAL = 0.2  # 200ms interval between predictions

# Object detection constants
RADAR_MIN_RANGE = 2.5  # meters (radar specs)
RADAR_MAX_RANGE = 30.0  # meters
EMERGENCY_RANGE = 5.0  # meters (stop when object is 5m or less)
MID_RANGE = 10.0  # meters (slow down when object is between 5m and 10m)

# Speed settings
EMERGENCY_SPEED = 0.0  # km/h (complete stop)
MEDIUM_SPEED = 10.0  # km/h (slow speed when object is at medium range)
NORMAL_SPEED = 20.0  # km/h (normal cruising speed)
MAX_SPEED = 40.0  # km/h

# Sensor configuration constants
FRONT_ANGLE_THRESHOLD = np.pi/4  # 45 degrees
LIDAR_MIN_RANGE = 0.2  # meters

# Control parameters
EMERGENCY_BRAKE_INTENSITY = 1.0
MEDIUM_BRAKE_INTENSITY = 0.4
MEDIUM_THROTTLE = 0.3
NORMAL_THROTTLE = 1.0

# set target speed
def set_speed(offset_speed):
    global speed
    
    speed = np.clip(speed + offset_speed, 0, 40)

    print(f"Set speed to {speed}")

#update steering angle
def set_steering_angle(wheel_angle):
    global angle, steering_angle
    # Check limits of steering
    if (wheel_angle - steering_angle) > 0.1:
        wheel_angle = steering_angle + 0.1
    if (wheel_angle - steering_angle) < -0.1:
        wheel_angle = steering_angle - 0.1
    steering_angle = wheel_angle
  
    # limit range of the steering angle
    if wheel_angle > 0.5:
        wheel_angle = 0.5
    elif wheel_angle < -0.5:
        wheel_angle = -0.5
    # update steering angle
    angle = wheel_angle

#validate increment of steering angle
def change_steer_angle(inc):
    global manual_steering
    # Apply increment
    new_manual_steering = manual_steering + inc
    # Validate interval 
    if new_manual_steering <= 25.0 and new_manual_steering >= -25.0: 
        manual_steering = new_manual_steering
        # Round to ensure we get exact multiples of 0.02
        set_steering_angle(round(manual_steering * 0.02, 2))
    # Debugging
    if manual_steering == 0:
        print("going straight")
    else:
        turn = "left" if steering_angle < 0 else "right"
        print("turning {} rad {}".format(str(steering_angle),turn))
        print("manual steering: ", manual_steering)

# Steering control constants
STEERING_INCREMENT = 0.02  # Fixed increment in radians per key press
MAX_STEERING_ANGLE = 0.5  # Maximum steering angle in radians
STEERING_SMOOTHING = 0.1  # Maximum change per timestep

# Steering state
target_steering = 0.0  # Target steering angle
current_steering = 0.0  # Current smoothed steering angle
prev_keys = {
    'LEFT': False,
    'RIGHT': False
}


def update_steering():
    global current_steering, target_steering
    # Smoothly interpolate current steering towards target
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
    # Solo se guardan los últimos 10 valores
    steering_angle_median_buffer.append(angle)
    if len(steering_angle_median_buffer) > buffer_size:
        steering_angle_median_buffer.pop(0)
    return np.median(steering_angle_median_buffer)


def perform_auto_driving(camera):
    global is_auto_driving, target_steering, last_prediction_time

    if not is_auto_driving:
        return
    
    # Check if enough time has passed since last prediction
    current_time = time.time()
    if current_time - last_prediction_time < PREDICTION_INTERVAL:
        return
        
    # Get image from camera
    image = get_image(camera)
    
    # Get roi image
    roi_image = get_roi_image(image)              # NumPy uint8 en [0,255]
    roi_image = roi_image.astype('float32') / 255.0  # normaliza a [0,1]

    # Convert image to tensor and RGB to TUV
    roi_image = tf.convert_to_tensor(roi_image, dtype=tf.float32)
    
    # roi_image tras rgb_to_yuv queda shape=(66,200,3)
    roi_image = tf.expand_dims(roi_image, axis=0)  # shape=(1,66,200,3)
    angle_pred = keras_model.predict(roi_image, verbose=0)    # devuelve array con shape (1,1)
    angle = float(angle_pred[0][0])                # extraes el valor escalar

    print(f"Predicted angle: {angle:.9f} rad")
    
    target_steering = np.clip(angle, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)
    
    # Update last prediction time
    last_prediction_time = current_time


def handle_steering_keys(key, keyboard):
    global target_steering, prev_keys
    
    # Detect key press events
    left_pressed = key == keyboard.LEFT and not prev_keys['LEFT']
    right_pressed = key == keyboard.RIGHT and not prev_keys['RIGHT']
    
    # Update key states
    prev_keys['LEFT'] = key == keyboard.LEFT
    prev_keys['RIGHT'] = key == keyboard.RIGHT
    
    # Apply steering changes only on press
    if left_pressed:
        target_steering = max(target_steering - STEERING_INCREMENT, -MAX_STEERING_ANGLE)
        print(f"Steering left: {target_steering:.2f} rad")
    elif right_pressed:
        target_steering = min(target_steering + STEERING_INCREMENT, MAX_STEERING_ANGLE)
        print(f"Steering right: {target_steering:.2f} rad")


def combined_lidar_radar_control(lidar, radar, current_speed, driver):
    # Get LIDAR data and apply filters
    lidar_ranges = lidar.getRangeImage()
    lidar_fov = lidar.getFov()  # Field of view in radians
    num_points = len(lidar_ranges)
    
    # Calculate angles for each point in the lidar scan
    angle_step = lidar_fov / num_points
    angles = np.array([i * angle_step - lidar_fov/2 for i in range(num_points)])
    
    # Filter for points in front of the car
    front_indices = np.where(np.abs(angles) <= FRONT_ANGLE_THRESHOLD)[0]
    
    # Debug information about LIDAR data
    print(f"\n=== LIDAR Debug Info ===")
    print(f"LIDAR FOV: {np.degrees(lidar_fov):.1f}°")
    print(f"Number of LIDAR points: {num_points}")
    print(f"Points in front cone: {len(front_indices)}")
    
    # Get ranges only in front cone and remove invalid readings
    front_ranges = []
    for idx in front_indices:
        range_val = lidar_ranges[idx]
        # Only consider finite readings within valid range
        if range_val != float('inf') and LIDAR_MIN_RANGE <= range_val <= RADAR_MAX_RANGE:
            front_ranges.append(range_val)
    
    # Print detailed LIDAR range information
    if front_ranges:
        print(f"Valid LIDAR readings: {len(front_ranges)} points")
        print(f"LIDAR ranges: {min(front_ranges):.2f}m to {max(front_ranges):.2f}m")
        # Print the 5 closest readings
        sorted_ranges = sorted(front_ranges)[:5]
        print(f"5 closest LIDAR points: {[f'{x:.2f}m' for x in sorted_ranges]}")
    else:
        print("No valid LIDAR readings in front cone")
    
    # Get minimum distance from filtered LIDAR data
    lidar_min_dist = min(front_ranges) if front_ranges else float('inf')

    # Get RADAR data and filter for frontal targets
    print(f"\n=== RADAR Debug Info ===")
    radar_targets = radar.getTargets()
    print(f"Total radar targets: {len(radar_targets)}")
    
    radar_dists = []
    for target in radar_targets:
        # Print each target's information
        print(f"Target: distance={target.distance:.2f}m, azimuth={np.degrees(target.azimuth):.1f}°")
        
        # Filter out likely false positives (exactly 1.00m is suspicious)
        if abs(target.distance - 1.00) < 0.01:
            print("Ignoring suspicious radar target at exactly 1.00m")
            continue
            
        # Only consider targets within the front cone and valid range
        if (abs(target.azimuth) <= FRONT_ANGLE_THRESHOLD and 
            RADAR_MIN_RANGE <= target.distance <= RADAR_MAX_RANGE):
            radar_dists.append(target.distance)
    
    if radar_dists:
        print(f"Valid radar targets: {len(radar_dists)}")
        print(f"Radar ranges: {min(radar_dists):.2f}m to {max(radar_dists):.2f}m")
    else:
        print("No valid radar targets in front cone")
    
    radar_min_dist = min(radar_dists) if radar_dists else float('inf')

    # Report the closest object
    min_dist = min(lidar_min_dist, radar_min_dist)
    print(f"\n📏 Closest object: LIDAR={lidar_min_dist:.2f}m, RADAR={radar_min_dist:.2f}m")

    # 🚨 Emergency: object is 5m or less
    if min_dist <= EMERGENCY_RANGE:
        # Additional validation for very close objects
        if min_dist == lidar_min_dist or len(radar_dists) > 0:  # Trust LIDAR or verified radar
            print(f"🟥 Emergency stop - object at {min_dist:.1f}m")
            driver.setThrottle(0.0)
            driver.setBrakeIntensity(EMERGENCY_BRAKE_INTENSITY)
            return EMERGENCY_SPEED

    # ⚠️ Object at medium distance (between 5m and 10m)
    if EMERGENCY_RANGE < min_dist <= MID_RANGE:
        print(f"⚠️ Object at medium range ({min_dist:.1f}m) - reducing speed to {MEDIUM_SPEED}km/h")
        driver.setBrakeIntensity(MEDIUM_BRAKE_INTENSITY)
        driver.setThrottle(MEDIUM_THROTTLE)
        return MEDIUM_SPEED

    # 🟢 Path is clear (object is more than 10m away or no object detected)
    print("🟢 Path clear - proceeding at normal speed")
    driver.setBrakeIntensity(0.0)
    driver.setThrottle(NORMAL_THROTTLE)
    return NORMAL_SPEED

# main
def main():
    global recording_name, up_down_pressed, space_pressed, is_auto_driving

    # check cuda compatible with tensorflow
    print(tf.config.list_physical_devices('GPU'))
    speed = 15  # o el valor inicial que tenías definido

    # Create the Robot instance.
    robot = Car()
    print("Robot conectado")

    driver = Driver()
    sup = Supervisor()

    # Get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())
    print("obtuvimos el timestep")

    # Create cameras instances

    # Center camera
    # Note: The camera is enabled with the timestep to ensure it captures images at the correct rate.
    # The camera's resolution is set to 640x480.
    # The camera's field of view is set to 90 degrees.
    # The camera's position is set to the center of the vehicle.
    # The camera's orientation is set to look forward.
    # The camera's name is "camera_center".
    camera = robot.getDevice("camera")
    print("obtuvimos la cámara")
    camera.enable(timestep)  # timestep

    lidar = robot.getDevice("front_lidar")
    lidar.enable(timestep)

    radar = robot.getDevice("front_radar")
    radar.enable(timestep)

    # processing display
    display_img = Display("display")
    print("display procesado")

    #create keyboard instance
    keyboard=Keyboard()
    keyboard.enable(timestep)
    print("habilitado el teclado")

    while robot.step() != -1:
        # Get image from camera
        image = get_image(camera)

        # Process and display image 
        display_image(display_img, image)

        # Read keyboard
        key = keyboard.getKey()

        if key == keyboard.UP:
            if not up_down_pressed:
                set_speed(5.0)
                print("up")
            up_down_pressed = True 

        elif key == keyboard.DOWN:
            if not up_down_pressed:
                set_speed(-5.0)
                print("down")
            up_down_pressed = True

        elif key == ord(' '):
            if not space_pressed:  # Only toggle if space wasn't pressed before
                is_auto_driving = not is_auto_driving
                print(f"Auto Driving: {'ON' if is_auto_driving else 'OFF'}")
            space_pressed = True
            
        elif space_pressed and key == -1:  # Space key released
            space_pressed = False

        elif up_down_pressed and key == -1:
            up_down_pressed = False

        else:
            # Handle steering with new key press detection
            handle_steering_keys(key, keyboard)
            
        # Update steering with smoothing
        angle = update_steering()
            
        #update angle and speed
        driver.setSteeringAngle(angle)
        speed = combined_lidar_radar_control(lidar, radar, speed, driver)
        driver.setCruisingSpeed(speed)
        perform_auto_driving(camera)

        # display steering angle in the top left corner
        # using the display of the supervisor
        sup.setLabel(
            0,
            f"Steering angle: {angle:.2f}",
            0.05,    # x = 5% from left
            0.85,    # y = 95% from bottom
            0.05,    # text height = 5% of screen
            0xFFFFFF,# white color
            0.0,     # fully opaque
            "Arial"  # font name
        )

        # display steering angle in the top left corner
        # using the display of the supervisor
        sup.setLabel(
            1,
            f"Speed: {speed} km/h",
            0.05,    # x = 5% from left
            0.90,    # y = 95% from bottom
            0.05,    # text height = 5% of screen
            0xFFFFFF,# white color
            0.0,     # fully opaque
            "Arial"  # font name
        )

        # display recording status below steering angle
        sup.setLabel(
            2,  # different ID for second label
            f"Auto Driving: {'ON' if is_auto_driving else 'OFF'}",
            0.05,    # x = 5% from left
            0.95,    # y = 90% from bottom (above steering angle)
            0.05,    # text height = 5% of screen
            0xFFFFFF,# white color
            0.0,     # fully opaque
            "Arial"  # font name
        )


if __name__ == "__main__":
    main()