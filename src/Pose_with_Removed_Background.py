#!/usr/bin/env python3

from PIL import Image, ImageDraw
import numpy as np
import cv2
from ultralytics import YOLO
import torch
from trials import get_ball_color , draw_3d_ball , get_line_color

# Define the keypoint connections
skeleton = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
    [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
    [1, 3], [2, 4], [3, 5], [4, 6]
]

def check_device():
    # Check if CUDA is available and return the appropriate device
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        return torch.device("cuda")
    else:
        print("CUDA is not available. Using CPU.")
        return torch.device("cpu")

def initialize_camera(camera_index=0, width=640, height=480):
    # Initialize the camera and set frame width and height
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video stream.")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def plot_keypoints_on_black_background(keypoints, width, height):
    """
    Plot keypoints on a black background with 3D-like balls.
    """
    line_width = 4
    ball_size = 11
    conf_thresh = 0.5
    width, height = 1024, 864


    # Create a black background
    black_background = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(black_background)

    # Define the color for the balls
    ball_color = get_ball_color()
    # Draw the keypoints with a 3D ball effect
    ball_size = 12
    for point in keypoints:
        if len(point) == 2:  # Ensure point has x and y coordinates
            x, y = int(point[0] * width), int(point[1] * height)
            draw_3d_ball(draw, (x, y), ball_size, ball_color)  # Draw 3D ball effect

    
    copied_keypoints = np.array(np.copy(keypoints[np.any(keypoints != 0, axis=1)]))
    copied_keypoints = np.hstack([copied_keypoints, np.ones((copied_keypoints.shape[0], 1))])

    while len(copied_keypoints) < 17:
        copied_keypoints = np.vstack([copied_keypoints, [0, 0, 0]])
        
    for pt1, pt2 in skeleton:
        if pt1 < len(copied_keypoints) and pt2 < len(copied_keypoints):
            if copied_keypoints[pt1][2] > conf_thresh and copied_keypoints[pt2][2] > conf_thresh:
                x1, y1 = int(copied_keypoints[pt1][0] * width), int(copied_keypoints[pt1][1] * height)
                x2, y2 = int(copied_keypoints[pt2][0] * width), int(copied_keypoints[pt2][1] * height)
                line_color = get_line_color(pt1, pt2)
                draw.line([x1, y1, x2, y2], fill=line_color, width=line_width)  # Thicker line for better visibility

    return black_background

def main():
    try:

        # Check the device and load the model
        device = check_device()
        model = YOLO("yolov8l-pose.pt").to(device)
        
        # Initialize the camera
        cap = initialize_camera(2, 1024, 720)
        
        # Loop to continuously get frames from the webcam
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break

            # Run YOLOv8 pose detection on the current frame
            results = model(frame)

            # Check if the keypoints attribute is present in the results
            if hasattr(results[0], 'keypoints'):
                # Access the keypoints for the first detected object
                keypoints = results[0].keypoints
                # Convert keypoints to numpy array and access the keypoints for the first detected object
                keypoints_numpy = keypoints.xyn.cpu().numpy()[0]
                black_background = plot_keypoints_on_black_background(keypoints_numpy, 1024, 720)
                
                print(keypoints_numpy)      
            else:
                print("No keypoints attribute found in the results.")
                black_background = Image.new("RGB", (1024, 720), (0, 0, 0))  # Black background if no keypoints

            # Plot the results on the frame
            plotted_frame = results[0].plot()

            # Convert PIL image to OpenCV format for displaying
            black_background_cv = np.array(black_background)

            # Display the frame with the plotted results and the keypoints on black background
            cv2.imshow("YOLOv8 Pose Detection", plotted_frame)
            cv2.imshow("Keypoints on Black Background", black_background_cv)

            # Break the loop if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Release the webcam and close the windows
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
