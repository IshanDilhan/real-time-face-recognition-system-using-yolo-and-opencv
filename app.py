from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import cv2
import numpy as np
import pickle
from deepface import DeepFace
from ultralytics import YOLO
from pymongo import MongoClient
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload paths
IMAGE_UPLOAD_FOLDER = "students/images"
VIDEO_UPLOAD_FOLDER = "students/videos"
ENCODINGS_FILE = "encodings.pkl"

app.config["UPLOAD_FOLDER"] = IMAGE_UPLOAD_FOLDER
app.config["VIDEO_FOLDER"] = VIDEO_UPLOAD_FOLDER

# Connect to MongoDB
##client = MongoClient("mongodb://localhost:27017/")
# db = client["facetrack"]
# attendance_collection = db["attendance"]

# Load YOLO Model
face_model = YOLO("model/best.pt")

# Load or compute student encodings
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        student_encodings = pickle.load(f)
else:
    student_encodings = {}

# Function to save encodings
def save_encodings():
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(student_encodings, f)

# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Upload Images
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        student_name = request.form["student_name"]
        student_folder = os.path.join(IMAGE_UPLOAD_FOLDER, student_name)
        os.makedirs(student_folder, exist_ok=True)

        images = request.files.getlist("images")
        for img in images:
            filename = secure_filename(img.filename)
            img.save(os.path.join(student_folder, filename))

        # Encode student images
        encodings = []
        for img_file in os.listdir(student_folder):
            img_path = os.path.join(student_folder, img_file)
            try:
                embedding = DeepFace.represent(img_path, model_name="Facenet")[0]["embedding"]
                encodings.append(np.array(embedding))
            except:
                print(f"Skipping {img_path} due to error")

        if encodings:
            student_encodings[student_name] = np.mean(encodings, axis=0)
            save_encodings()

        return redirect(url_for("home"))

    return render_template("upload.html")

# Upload Video
@app.route("/upload_video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return "No video file found", 400
    
    video = request.files["video"]
    filename = secure_filename(video.filename)
    video_path = os.path.join(VIDEO_UPLOAD_FOLDER, filename)
    video.save(video_path)

    return redirect(url_for("detect_video", filename=filename))

# Detect Faces from Video
@app.route("/detect_video/<filename>")
def detect_video(filename):
    video_path = os.path.join(VIDEO_UPLOAD_FOLDER, filename)
    cap = cv2.VideoCapture(video_path)
    detected_students = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = face_model(frame)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_crop = frame[y1:y2, x1:x2]

                try:
                    face_embedding = DeepFace.represent(face_crop, model_name="Facenet")[0]["embedding"]
                    face_embedding = np.array(face_embedding)

                    recognized_student = "Unknown"
                    min_distance = float("inf")

                    for student, encoding in student_encodings.items():
                        similarity = np.linalg.norm(encoding - face_embedding)
                        if similarity < min_distance and similarity < 10:
                            min_distance = similarity
                            recognized_student = student

                    detected_students.add(recognized_student)

                except:
                    continue

    cap.release()

    # Store in MongoDB
    for student in detected_students:
        attendance_collection.insert_one({"student": student})

    return redirect(url_for("attendance"))

# View Attendance
@app.route("/attendance")
def attendance():
    records = attendance_collection.find()
    return render_template("results.html", records=records)

if __name__ == "__main__":
    app.run(debug=True)
