# Pose_Interaction
This repository demonstrates the use of **YOLO-Pose** for human pose estimation on various input types, including video streams, images, and video files. The implementation also visualizes the human pose on a black background for enhanced clarity.


## **Features**
1. **Video Stream Pose Estimation**
   - YOLO-Pose is applied in real-time on video streams.
   - Outputs human pose points visualized over a black background.

2. **Pose Estimation on Images**
   - Takes individual images as input.
   - Processes the images using YOLO-Pose and outputs the pose over a black background.

3. **Pose Estimation on Videos**
   - Processes video files frame by frame.
   - Outputs a video file showing human pose on a black background.

4. **Real-Time Stream Pose Estimation**
   - Implements real-time pose estimation on streaming data.
   - Displays the human pose on a black background in real time.

---

## **Technologies Used**
- **YOLO-Pose**: For robust human pose detection.
- **OpenCV**: For handling video streams and image processing.
- **PyTorch**: For deep learning inference using YOLO-Pose.
- **Python**: Core programming language for the pipeline.

---
## **Files in src**

- [test.ipynb] 
Used to make all the deployment of yolo and some tests, usually i use it  to just make sure that everything is ok before i use it in python file code.

- [pose.py] 
The file in which i use the deployed model with the normal camera stream and show the results.

- [pose_with_removed_back_photo.py] 
The first trial to make pose detection and then try to make it on a black background like i subtract the keypoint from the original scene and visulize it on black background.

- pose_with_removed_back_video.py 
The same as the above but on a video.

- [Pose_with_Removed_Background.py] 
The most important file as it's the file i use to show the real life streaming pose on a black background which in my opinion can be used as interactive solution.

- [helping_functions.py] 
All the functions that i could use in several python files in there in that file, as it's like header file in C languange.

---

## **Results**

### **Normal Pose Detection**
| Original Image | Pose on Black Background |
|----------------|---------------------------|
| ![Original Image](/photos/12.jpg) | ![Pose Image](/results/12.jpg) |

### **Pose on Black Background (Image)**
| Original Image and Pose on Black Background |
|-------------------------------------------|
| ![Both images](/results/output_with_keypoints.png)|

### **Pose on Black Background (Video)**
#### **Input Video**
![Input Video](/video_result/video.mp4)

#### **Output Video**
![Output Video](/video_result/output_with_keypoints.avi)

### **Pose on From Stream (Video)**
![Output Video](/realtime_video_stream_output/Pose.mp4)
