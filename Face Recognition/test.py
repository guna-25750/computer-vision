import cv2
import numpy as np
import os

# Path to Haar Cascade for face detection
haar_file = 'haarcascade_frontalface_default.xml'
dataset = 'datasets'
print('Training...')

# Prepare data containers
images, labels, names = [], [], {}
current_id = 0

# Loop through dataset directories; each subdirectory is a person
for subdir in os.listdir(dataset):
    subjectpath = os.path.join(dataset, subdir)
    if not os.path.isdir(subjectpath):
        continue
    names[current_id] = subdir
    for filename in os.listdir(subjectpath):
        path = os.path.join(subjectpath, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # Improve contrast using histogram equalization
        img = cv2.equalizeHist(img)
        # Resize training images for consistency
        img = cv2.resize(img, (130, 100))
        images.append(img)
        labels.append(current_id)
    current_id += 1

images = np.array(images)
labels = np.array(labels)

# Create LBPH face recognizer with custom parameters for better accuracy
model = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8, grid_x=8, grid_y=8)

# Define a threshold for recognition confidence (lower means stricter matching)
THRESHOLD = 800  

# Try to use setThreshold if supported; if not, use manual thresholding later.
try:
    model.setThreshold(THRESHOLD)
except Exception as e:
    print("setThreshold not supported; using manual thresholding.")

# Train the model with the collected images and labels
model.train(images, labels)

# Initialize face detector using Haar cascade
face_cascade = cv2.CascadeClassifier(haar_file)
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Convert frame to grayscale and enhance contrast
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Detect faces in the frame with tuned parameters
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(100, 100))
    
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        try:
            face_resized = cv2.resize(face, (130, 100))
        except Exception as e:
            continue
        
        # (Optional) Equalize the ROI to ensure consistent preprocessing
        face_resized = cv2.equalizeHist(face_resized)

        # Predict the face label and confidence
        prediction = model.predict(face_resized)
        
        # If the prediction confidence is high or label is -1, mark as Unknown
        if prediction[0] == -1 or prediction[1] > THRESHOLD:
            label_text = "Unknown"
        else:
            label_text = f"{names[prediction[0]]}"
        
        # Draw rectangle and label on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label_text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (51, 250, 250), 2)
    
    cv2.imshow('Face Recognition', frame)
    key = cv2.waitKey(10)
    if key == ord('p'):  # Press 'p' to exit
        break

cam.release()
cv2.destroyAllWindows()
