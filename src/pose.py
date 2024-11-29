#!/usr/bin/env python3

import cv2
from ultralytics import YOLO
import torch

def check_device():
    # Check if CUDA is available and return the appropriate device
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        return torch.device("cuda")
    else:
        print("CUDA is not available. Using CPU.")
        return torch.device("cpu")

def initialize_camera(camera_index=1, width=640, height=480):
    # Initialize the camera and set frame width and height
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video stream.")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

def main():
    try:
        # Check the device and load the model
        device = check_device()
        model = YOLO("yolov8l-pose.pt").to(device)
        
        # Initialize the camera
        cap = initialize_camera(0, 640, 480)
        
        # Loop to continuously get frames from the webcam
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break

            # Run YOLOv8 pose detection on the current frame
            results = model(frame)

            # Plot the results on the frame
            plotted_frame = results[0].plot()

            # Display the frame with the plotted results
            cv2.imshow("YOLOv8 Pose Detection", plotted_frame)

            # Break the loop if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Release the webcam and close the window
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
