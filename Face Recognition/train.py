import cv2
import os

# Constants
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
datasets = 'datasets'

# Get user input for subject name and sanitize it
subsets = input("Enter person's name: ").strip().replace(" ", "_")  # Remove spaces
path = os.path.join(datasets, subsets)

# Create directory if not exists
if not os.path.isdir(path):
    os.makedirs(path, exist_ok=True)

# Face detection parameters
(width, height) = (130, 100)
face_cascade = cv2.CascadeClassifier(haar_file)
cam = cv2.VideoCapture(0)

# Image capture parameters
count = 1
total_images = 100  # Total images to capture
wait_time = 50  # ms between frames

# Ensure camera is opened
if not cam.isOpened():
    print("Error: Camera not accessible")
    exit()

try:
    while count <= total_images:
        ret, img = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Improved face detection parameters
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,  # More careful scaling
            minNeighbors=5,   # Reduce false positives
            minSize=(30, 30) ) # Minimum face size
        
        # Only save if exactly 1 face detected
        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            
            # Extract and resize face
            face_roi = gray[y:y+h, x:x+w]
            if face_roi.size == 0:
                continue  # Skip empty frames
                
            face_resized = cv2.resize(face_roi, (width, height))
            
            # Save image with leading zeros
            cv2.imwrite(f"{path}/{subsets}_{count:03d}.jpg", face_resized)
            
            # Draw visual feedback
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(img, f"Capturing: {count}/{total_images}", (10,30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            count += 1
        else:
            cv2.putText(img, "Align face in frame", (10,30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow('Face Data Collection', img)
        key = cv2.waitKey(wait_time)
        if key in [ord('q'), 27]:  # Exit on Q or ESC
            break

finally:
    cam.release()
    cv2.destroyAllWindows()
    print(f"Completed! Saved {count-1} images in {path}")