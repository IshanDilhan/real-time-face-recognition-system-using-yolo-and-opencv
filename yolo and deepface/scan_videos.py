from ultralytics import YOLO
import cv2
import numpy as np
from face_verification import DeepFace
import os

# Load the YOLO face detection model
face_model = YOLO("model/best.pt")  # Update with your trained model

# Path to students' images
STUDENT_IMAGES_PATH = "students/images"
student_encodings = {}

# Function to encode all student images and store averaged encodings
def encode_student_faces():
    for student_folder in os.listdir(STUDENT_IMAGES_PATH):
        student_path = os.path.join(STUDENT_IMAGES_PATH, student_folder)
        if os.path.isdir(student_path):  # Ensure it's a folder
            encodings = []
            for img_name in os.listdir(student_path):
                img_path = os.path.join(student_path, img_name)
                try:
                    embedding = DeepFace.represent(img_path, model_name="Facenet")[0]['embedding']
                    encodings.append(np.array(embedding))
                except:
                    print(f"Skipping {img_path} due to error")
            
            if encodings:
                student_encodings[student_folder] = np.mean(encodings, axis=0)  # Average encoding

# Encode student faces before processing the video
encode_student_faces()

# Load video instead of webcam
VIDEO_PATH = "students/videos/7/3.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = face_model(frame)  # Detect faces

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            face_crop = frame[y1:y2, x1:x2]  # Crop detected face

            try:
                face_embedding = DeepFace.represent(face_crop, model_name="Facenet")[0]['embedding']
                face_embedding = np.array(face_embedding)

                # Compare detected face with stored encodings
                recognized_student = "Unknown"
                min_distance = float("inf")

                for student, encoding in student_encodings.items():
                    similarity = np.linalg.norm(encoding - face_embedding)

                    if similarity < min_distance and similarity < 10:  # Threshold
                        min_distance = similarity
                        recognized_student = student  # Save recognized name
                
                print(f"Detected: {recognized_student}")  # Print student name
                
                cv2.putText(frame, recognized_student, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            except:
                pass  # Ignore errors in face embedding

    cv2.imshow("Face Recognition - Video", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
