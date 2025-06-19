from controller import Display, Keyboard, Robot, Camera, Supervisor
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os
import time
import csv


BASE_DIR = "./datasets"


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
    # Get image dimensions and crop bottom half
    h, w, _ = image.shape
    cropped = image[h//2:h, 0:w]  # Take bottom half (80 rows)
    
    # Resize to target dimensions using area interpolation
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
is_recording = False  # Flag to track recording status
recording_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
last_recording_time = time.time()
space_pressed = False
up_down_pressed = False

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


def start_recording():
    global is_recording, recording_name

    if not is_recording:
        return
    
    recording_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    # create a folder to store the images. The folder name is timestamp
    folder_name = f"{BASE_DIR}/{recording_name}"
    os.makedirs(folder_name, exist_ok=True)
    
    # CSV file name
    csv_file_name = f"{BASE_DIR}/{recording_name}.csv"
    # create a CSV file to store the data
    csv_file = open(csv_file_name, "w")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["left_image", "center_image", "right_image", "steering_angle", "timestamp"])
    csv_file.close()


def perform_recording(center_camera, left_camera, right_camera, angle):
    global is_recording, recording_name, last_recording_time
    if not is_recording:
        return
    
    # only record every 200ms
    if time.time() - last_recording_time < 0.2:
        return
    
    last_recording_time = time.time()
    
    # get timestamp
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

    # get the image from the camera
    image = get_image(center_camera)
    # get the image from the camera left
    image_left = get_image(left_camera)
    # get the image from the camera right
    image_right = get_image(right_camera)

    # save only the roi image
    roi_image_center = get_roi_image(image)
    roi_image_left = get_roi_image(image_left)
    roi_image_right = get_roi_image(image_right)

    # change to BGR before saving
    roi_image_center = cv2.cvtColor(roi_image_center, cv2.COLOR_RGB2BGR)
    roi_image_left = cv2.cvtColor(roi_image_left, cv2.COLOR_RGB2BGR)
    roi_image_right = cv2.cvtColor(roi_image_right, cv2.COLOR_RGB2BGR)

    # image paths
    image_center_path = f"{recording_name}/{timestamp}_center_image.jpg"
    image_left_path = f"{recording_name}/{timestamp}_left_image.jpg"
    image_right_path = f"{recording_name}/{timestamp}_right_image.jpg"
    
    # save the roi images
    cv2.imwrite(BASE_DIR + "/" + image_center_path, roi_image_center)
    cv2.imwrite(BASE_DIR + "/" + image_left_path, roi_image_left)
    cv2.imwrite(BASE_DIR + "/" + image_right_path, roi_image_right)

    # save the data to the CSV file
    csv_file = open(f"{BASE_DIR}/{recording_name}.csv", "a")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([image_center_path, image_left_path, image_right_path, f"{angle:.4f}", timestamp])
    csv_file.close()

    print(f"Recording at {timestamp}")


# main
def main():
    global recording_name, up_down_pressed, space_pressed, is_recording

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
    camera = robot.getDevice("camera_center")
    camera.enable(timestep)  # timestep

    # set camera left and right
    camera_left = robot.getDevice("camera_left")
    camera_left.enable(timestep)  # timestep
    camera_right = robot.getDevice("camera_right")
    camera_right.enable(timestep)  # timestep

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
                is_recording = not is_recording

                # start recording
                start_recording()

                print(f"Recording: {'ON' if is_recording else 'OFF'}")
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

        perform_recording(camera, camera_left, camera_right, angle)

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
            f"Recording: {'ON' if is_recording else 'OFF'}",
            0.05,    # x = 5% from left
            0.95,    # y = 90% from bottom (above steering angle)
            0.05,    # text height = 5% of screen
            0xFFFFFF,# white color
            0.0,     # fully opaque
            "Arial"  # font name
        )


if __name__ == "__main__":
    main()