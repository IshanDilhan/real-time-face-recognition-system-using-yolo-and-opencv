from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import os
import cv2
import numpy as np
import pickle
import shutil
from deepface import DeepFace
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload paths
IMAGE_UPLOAD_FOLDER = "students/images"
VIDEO_UPLOAD_FOLDER = "students/videos"
ENCODINGS_FILE = "encodings.pkl"
STUDENT_IMAGE_DIR = "students/images"
app.config["UPLOAD_FOLDER"] = IMAGE_UPLOAD_FOLDER
app.config["VIDEO_FOLDER"] = VIDEO_UPLOAD_FOLDER

# Load YOLO Model
face_model = YOLO("model/best.pt")

# Load or compute student encodings
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        student_encodings = pickle.load(f)
    print("✅ Loaded existing student encodings.")
else:
    student_encodings = {}

# Function to save encodings
def save_encodings():
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(student_encodings, f)

# 🔹 Fetch Student List
@app.route('/get_students', methods=['GET'])
def get_students():
    students = [folder for folder in os.listdir(IMAGE_UPLOAD_FOLDER) if os.path.isdir(os.path.join(IMAGE_UPLOAD_FOLDER, folder))]
    return jsonify(students)

# 🔹 Serve Student Images
@app.route('/students/images/<student>/<filename>')
def serve_image(student, filename):
    student_folder = os.path.join(IMAGE_UPLOAD_FOLDER, student)  # Construct the correct folder path
    return send_from_directory(student_folder, filename)  # ✅ Correct usage

# 🔹 Fetch Images of a Selected Student
@app.route('/get_images/<student_name>', methods=['GET'])
def get_images(student_name):
    student_folder = os.path.join(IMAGE_UPLOAD_FOLDER, student_name)
    if os.path.exists(student_folder):
        images = os.listdir(student_folder)
        return jsonify(images)
    return jsonify([])

# 🔹 Add a New Student (Create Folder)
@app.route('/add_student', methods=['POST'])
def add_student():
    student_name = request.form['student_name']
    student_folder = os.path.join(IMAGE_UPLOAD_FOLDER, student_name)

    if not os.path.exists(student_folder):
        os.makedirs(student_folder)
        return jsonify({"status": "success", "message": "Student folder created."})
    else:
        return jsonify({"status": "error", "message": "Student already exists."})

# 🔹 Upload Images for a Student
@app.route("/upload_image", methods=["POST"])
def upload_image():
    student_name = request.form.get("student_name")
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "No selected file"}), 400

    student_path = os.path.join(STUDENT_IMAGE_DIR, student_name)
    if not os.path.exists(student_path):
        return jsonify({"status": "error", "message": "Student folder does not exist"}), 400

    file_path = os.path.join(student_path, file.filename)
    file.save(file_path)

    return jsonify({"status": "success"})

# 🔹 Delete an Image from a Student Folder
@app.route('/delete_image', methods=['POST'])
def delete_image():
    student_name = request.form['student_name']
    image_name = request.form['image_name']
    image_path = os.path.join(IMAGE_UPLOAD_FOLDER, student_name, image_name)

    if os.path.exists(image_path):
        os.remove(image_path)
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'File not found'})

# 🔹 Delete a Student Folder
@app.route('/delete_folder', methods=['POST'])
def delete_folder():
    student_name = request.form['student_name']
    student_folder = os.path.join(IMAGE_UPLOAD_FOLDER, student_name)

    if os.path.exists(student_folder):
        shutil.rmtree(student_folder)
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'Folder not found'})

# 🔹 Recognize Faces in a Video
@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'status': 'error', 'message': 'No video file provided'})

    video = request.files['video']
    video_path = os.path.join(VIDEO_UPLOAD_FOLDER, secure_filename(video.filename))
    os.makedirs(VIDEO_UPLOAD_FOLDER, exist_ok=True)
    video.save(video_path)

    cap = cv2.VideoCapture(video_path)
    detected_students = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces using YOLO
        results = face_model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_img = frame[y1:y2, x1:x2]

                # Recognize face using DeepFace
                try:
                    recognized = DeepFace.find(face_img, db_path=IMAGE_UPLOAD_FOLDER, enforce_detection=False)
                    if recognized and len(recognized) > 0:
                        student_name = recognized[0]['identity'][0].split('/')[-2]  # Extract student name
                        if student_name not in detected_students:
                            detected_students.append(student_name)
                except:
                    pass

    cap.release()
    return jsonify({'status': 'success', 'students_detected': detected_students})

# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# View Attendance (Dummy Page since MongoDB is commented)
@app.route("/attendance")
def attendance():
    # If MongoDB is used, uncomment the next line
    # records = attendance_collection.find()
    records = []  # Empty records since MongoDB is disabled
    return render_template("results.html", records=records)

# Upload Images Page
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

        return redirect(url_for("home"))

    return render_template("upload.html")
app.route("/add_student", methods=["POST"])
def add_student():
    student_name = request.form.get("student_name")
    if not student_name:
        return jsonify({"status": "error", "message": "Student name required"}), 400

    student_path = os.path.join(STUDENT_IMAGE_DIR, student_name)
    try:
        os.makedirs(student_path, exist_ok=True)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
