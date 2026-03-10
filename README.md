# face_detection

This Drowsiness Detection System is a real-time computer vision tool built primarily with OpenCV, which serves as the core library for frame processing and object detection via Haar Cascade classifiers. The system works by isolating the face region from a live camera feed and searching for eyes within that area; if eyes are not detected for a consecutive duration exceeding the defined threshold, a non-blocking alarm is triggered using Pygame and threading to ensure the video stream remains fluid.

Key Highlights:

OpenCV: Handles real-time image capture, preprocessing (grayscale conversion), and feature detection.

Haar Cascade Classifiers: Uses pre-trained XML models to efficiently identify facial and ocular landmarks.

Pygame: Manages the audio subsystem for the alarm, integrated with threading to prevent UI freezing during sound playback.

Logic Thresholding: Implements a timer-based state machine that distinguishes between brief, normal blinks and sustained eye closure.
