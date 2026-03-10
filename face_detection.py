import cv2 as cv  # type: ignore
import time
import pygame
import threading

# Load Haarcascade classifiers
face_cap = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cap = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

# Check if classifiers were loaded successfully
if face_cap.empty() or eye_cap.empty():
    print("Error: Failed to load Haarcascade XML files.")
    exit()

# Initialize pygame mixer for alarm sound
pygame.mixer.init()
ALARM_SOUND = "project/alarm.wav"

# Function to play alarm sound
def sound_alarm():
    if not pygame.mixer.music.get_busy():  # Prevent multiple overlapping sounds
        pygame.mixer.music.load(ALARM_SOUND)
        pygame.mixer.music.play()

# Capture video from the default camera
video_capture = cv.VideoCapture(0)

# Check if video capture is working
if not video_capture.isOpened():
    print("Error: Could not access the camera.")
    exit()

eyes_closed_time = None  # Initialize as None (not tracking yet)
ALARM_THRESHOLD = 3  # Time in seconds before triggering alarm
alarm_on = False  # Alarm should start as False

while True:
    # Read a frame
    ret, img = video_capture.read()

    if not ret:
        print("Failed to read frame from video capture")
        break

    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cap.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        # Detect eyes within the face
        eyes = eye_cap.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))

        if len(eyes) == 0:
            if eyes_closed_time is None:
                eyes_closed_time = time.time()  # Start timer
            elif time.time() - eyes_closed_time >= ALARM_THRESHOLD:
                if not alarm_on:
                    alarm_on = True
                    t = threading.Thread(target=sound_alarm)
                    t.daemon = True
                    t.start()
        else:
            eyes_closed_time = None  # Reset timer
            alarm_on = False
            pygame.mixer.music.stop()

        # Draw eye rectangles
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

    # Show frame
    cv.imshow("Drowsiness Detection", img)

    # Exit on 'q' key press
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_capture.release()
cv.destroyAllWindows()
pygame.mixer.quit()
