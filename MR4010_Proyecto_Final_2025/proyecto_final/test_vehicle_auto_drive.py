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


MODEL_PATH = "./models/best_model_cpu_v4.h5"
MODEL_PATH_2 = "./models/best_model_vgg_cpu.h5"
MODEL_TO_USE = "vgg16"


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

if MODEL_TO_USE == "vgg16":
    # add class PadAndResize to the model
    with tf.keras.utils.custom_object_scope({'PadAndResize': PadAndResize}):
        keras_model = load_model(MODEL_PATH_2)
else:
    with tf.device('/CPU:0'):
        keras_model = load_model(MODEL_PATH)

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
    global is_auto_driving, target_steering

    if not is_auto_driving:
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


# main
def main():
    global recording_name, up_down_pressed, space_pressed, is_auto_driving

    # Create the Robot instance.
    robot = Car()
    driver = Driver()
    sup = Supervisor()

    # Get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())

    # Create cameras instances

    # Center camera
    # Note: The camera is enabled with the timestep to ensure it captures images at the correct rate.
    # The camera's resolution is set to 640x480.
    # The camera's field of view is set to 90 degrees.
    # The camera's position is set to the center of the vehicle.
    # The camera's orientation is set to look forward.
    # The camera's name is "camera_center".
    camera = robot.getDevice("camera")
    camera.enable(timestep)  # timestep

    # processing display
    display_img = Display("display")

    #create keyboard instance
    keyboard=Keyboard()
    keyboard.enable(timestep)

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