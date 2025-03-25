from ultralytics import YOLO
import cv2
import numpy as np
from deepface import DeepFace
import os
import pickle
import requests  # Handle the ESP32 stream

# Load the YOLO face detection model
face_model = YOLO("../model/best.pt")  # Update with your trained model

# Path to student images and encoded data
STUDENT_IMAGES_PATH = "../students/images"
ENCODINGS_FILE = "../student_encodings.pkl"

# Dictionary to store student encodings
student_encodings = {}

# Function to encode all student images and store averaged encodings
def encode_student_faces():
    global student_encodings
    if os.path.exists(ENCODINGS_FILE):  # Load precomputed encodings
        with open(ENCODINGS_FILE, "rb") as f:
            student_encodings = pickle.load(f)
        print("✅ Loaded existing student encodings.")
        return

    print("🔄 Encoding student images...")
    for student_folder in os.listdir(STUDENT_IMAGES_PATH):  # Iterate through student folders
        student_path = os.path.join(STUDENT_IMAGES_PATH, student_folder)
        
        if os.path.isdir(student_path):  # Ensure it's a folder
            encodings = []
            for img_name in os.listdir(student_path):  # Loop through images per student
                img_path = os.path.join(student_path, img_name)
                try:
                    embedding = DeepFace.represent(img_path, model_name="Facenet")[0]['embedding']
                    encodings.append(np.array(embedding))
                except Exception as e:
                    print(f"⚠️ Skipping {img_path} due to error: {e}")

            if encodings:  # Ensure at least one encoding is present
                student_encodings[student_folder] = np.mean(encodings, axis=0)  # Average encoding
                print(f"✅ Encoded: {student_folder}")

    # Save encodings for future use
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(student_encodings, f)
    print("✅ Encodings saved successfully!")

# Encode student faces before starting recognition
encode_student_faces()

# ESP32-CAM Stream URL (Update if needed)
ESP32_STREAM_URL = "http://192.168.8.164:81/stream"

# Function to get frames from the ESP32-CAM MJPEG stream
def get_esp32_frame():
    stream = requests.get(ESP32_STREAM_URL, stream=True)
    byte_data = b""
    
    for chunk in stream.iter_content(chunk_size=1024):
        byte_data += chunk
        a = byte_data.find(b'\xff\xd8')  # Start of JPEG
        b = byte_data.find(b'\xff\xd9')  # End of JPEG
        
        if a != -1 and b != -1:
            jpg = byte_data[a:b+2]
            byte_data = byte_data[b+2:]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            return frame

while True:
    frame = get_esp32_frame()
    if frame is None:
        print("❌ Failed to retrieve frame")
        continue

    # Resize frame for better processing speed
    frame = cv2.resize(frame, (640, 480))

    results = face_model(frame)  # Detect faces

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            face_crop = frame[y1:y2, x1:x2]  # Crop detected face

            # Ensure the cropped face is valid
            if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
                continue

            try:
                # Convert frame crop to RGB (OpenCV uses BGR)
                face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

                # Get embedding for detected face
                face_embedding = DeepFace.represent(face_crop_rgb, model_name="Facenet", enforce_detection=False)[0]['embedding']
                face_embedding = np.array(face_embedding)

                # Compare detected face with stored encodings
                recognized_student = "Unknown"
                min_distance = float("inf")

                for student, encoding in student_encodings.items():
                    similarity = np.linalg.norm(encoding - face_embedding)

                    if similarity < min_distance and similarity < 10:  # Threshold
                        min_distance = similarity
                        recognized_student = student  # Save recognized name
                
                print(f"🎓 Detected: {recognized_student}")  # Print student name
                
                cv2.putText(frame, recognized_student, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            except Exception as e:
                print(f"⚠️ Error processing face: {e}")

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cv2.destroyAllWindows()