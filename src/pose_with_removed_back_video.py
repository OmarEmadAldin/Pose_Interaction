#!/usr/bin/env python3

from PIL import Image, ImageDraw
import numpy as np
import cv2
from ultralytics import YOLO
import torch
from trials import get_ball_color, draw_3d_ball, get_line_color

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

def plot_keypoints_on_black_background(keypoints, width, height):
    """
    Plot keypoints on a black background with 3D-like balls.
    """
    line_width = 4
    ball_size = 11
    conf_thresh = 0.5

    # Create a black background with the specified width and height
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

def process_video(video_path, output_path):
    try:
        # Check the device and load the model
        device = check_device()
        model = YOLO("yolov8l-pose.pt").to(device)

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file at path: {video_path}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame to the expected input size (if necessary)
            resized_frame = cv2.resize(frame, (width, height))

            # Run YOLOv8 pose detection on the frame
            results = model(resized_frame)

            # Check if the keypoints attribute is present in the results
            if hasattr(results[0], 'keypoints'):
                # Access the keypoints for the first detected object
                keypoints = results[0].keypoints
                # Convert keypoints to numpy array and access the keypoints for the first detected object
                keypoints_numpy = keypoints.xyn.cpu().numpy()[0]
                black_background = plot_keypoints_on_black_background(keypoints_numpy, width, height)
            else:
                print("No keypoints attribute found in the results.")
                black_background = Image.new("RGB", (width, height), (0, 0, 0))  # Black background if no keypoints

            # Convert PIL image to OpenCV format for displaying
            black_background_cv = np.array(black_background)

            # Stack the original and the processed images side by side
            side_by_side = np.hstack((resized_frame, black_background_cv))

            # Write the frame to the output video
            out.write(side_by_side)

            # Display the result
            cv2.imshow("Original Frame and Keypoints", side_by_side)

            # Wait for key press to close the window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Ensure all windows are closed properly
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Process a specific video file
    video_path = "/home/omar_ben_emad/State of The Art/YOLO Pose/video.mp4"  # Replace with the path to your video
    output_path = "output_with_keypoints.avi"  # Replace with the desired output path
    process_video(video_path, output_path)
