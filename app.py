from flask import Flask, render_template, Response,request, redirect, url_for, jsonify, send_from_directory
import os
import cv2
import numpy as np
import pickle
import shutil
import threading
from deepface import DeepFace
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload paths
IMAGE_UPLOAD_FOLDER = "students/images"
VIDEO_UPLOAD_FOLDER = "students/videos"
STUDENT_IMAGE_DIR = "students/images"
STUDENT_IMAGES_PATH = "students/images"
ENCODINGS_FILE = "student_encodings.pkl"
UPLOAD_FOLDER = "students/videos"  # Set your correct directory
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER  # Store in app config
app.config["UPLOAD_FOLDER"] = IMAGE_UPLOAD_FOLDER
app.config["VIDEO_FOLDER"] = VIDEO_UPLOAD_FOLDER
stop_detection = False
# Load YOLO Model
face_model = YOLO("model/best.pt")

def encode_student_faces():
    """Encodes student images and saves embeddings."""
    global student_encodings
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            student_encodings = pickle.load(f)
        print("✅ Loaded existing student encodings.")
        return

    print("🔄 Encoding student images...")
    for student_folder in os.listdir(STUDENT_IMAGES_PATH):
        student_path = os.path.join(STUDENT_IMAGES_PATH, student_folder)
        if os.path.isdir(student_path):
            encodings = []
            for img_name in os.listdir(student_path):
                img_path = os.path.join(student_path, img_name)
                try:
                    embedding = DeepFace.represent(img_path, model_name="Facenet")[0]['embedding']
                    encodings.append(np.array(embedding))
                except Exception as e:
                    print(f"⚠️ Skipping {img_path}: {e}")

            if encodings:
                student_encodings[student_folder] = np.mean(encodings, axis=0)
                print(f"✅ Encoded: {student_folder}")

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(student_encodings, f)
    print("✅ Encodings saved!")

encode_student_faces()  # Load student encodings



# ---------------- 2️⃣ Upload Video & Detect Faces ----------------
@app.route("/upload_video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["video"]
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Return video stream endpoint
    return jsonify({"video_url": "/video_feed?file=" + filename})


def process_video(video_path):  # Ensure function accepts video_path
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = face_model(frame)  # Detect faces
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_crop = frame[y1:y2, x1:x2]

                try:
                    face_embedding = DeepFace.represent(face_crop, model_name="Facenet")[0]['embedding']
                    face_embedding = np.array(face_embedding)

                    recognized_student = "Unknown"
                    min_distance = float("inf")

                    for student, encoding in student_encodings.items():
                        similarity = np.linalg.norm(encoding - face_embedding)

                        if similarity < min_distance and similarity < 10:
                            min_distance = similarity
                            recognized_student = student  
                    
                    cv2.putText(frame, recognized_student, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                except:
                    pass  

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield frame for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()



@app.route("/video_feed")
def video_feed():
    video_file = request.args.get("file")
    
    if not video_file:
        print("❌ ERROR: No file parameter provided in request.")
        return jsonify({"error": "No file specified"}), 400

    video_path = os.path.join(app.config["UPLOAD_FOLDER"], video_file)
    
    if not os.path.exists(video_path):
        print(f"❌ ERROR: File '{video_path}' not found.")
        return jsonify({"error": "File not found"}), 404  # Handle missing files properly

    print(f"✅ Processing Video: {video_path}")  # Debugging log

    try:
        return Response(process_video(video_path), mimetype="multipart/x-mixed-replace; boundary=frame")
    except Exception as e:
        print(f"❌ ERROR in video_feed(): {str(e)}")  # Debugging log
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500




# ---------------- 3️⃣ Real-Time Face Detection ----------------
@app.route('/detect_realtime')
def detect_realtime():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------- 4️⃣ Process Uploaded Video ----------------
def process_uploaded_video(video_path):  # ✅ Renamed function
    """Detects faces in uploaded video and recognizes students."""
    cap = cv2.VideoCapture(video_path)

    output_path = f"static/detected_{os.path.basename(video_path)}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = face_model(frame)  # Detect faces
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_crop = frame[y1:y2, x1:x2]

                if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
                    continue

                try:
                    face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    face_embedding = DeepFace.represent(face_crop_rgb, model_name="Facenet", enforce_detection=False)[0]['embedding']
                    face_embedding = np.array(face_embedding)

                    recognized_student = "Unknown"
                    min_distance = float("inf")

                    for student, encoding in student_encodings.items():
                        similarity = np.linalg.norm(encoding - face_embedding)
                        if similarity < min_distance and similarity < 10:
                            min_distance = similarity
                            recognized_student = student
                
                    cv2.putText(frame, recognized_student, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                except Exception as e:
                    print(f"⚠️ Error processing face: {e}")

        out.write(frame)

    cap.release()
    out.release()
    return output_path



# ---------------- 5️⃣ Real-Time Face Recognition ----------------


@app.route('/start_realtime')
def start_realtime():
    global stop_detection
    stop_detection = False  # Reset stop flag when starting
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_realtime')
def stop_realtime():
    global stop_detection
    stop_detection = True  # Set stop flag
    return jsonify({"message": "Real-time detection stopped"})

def generate_frames():
    global stop_detection
    cap = cv2.VideoCapture(0)  # Open Webcam

    while True:
        if stop_detection:  # Check if stop is requested
            break

        success, frame = cap.read()
        if not success:
            break

        results = face_model(frame)  # Detect faces
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_crop = frame[y1:y2, x1:x2]

                if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
                    continue

                try:
                    face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    face_embedding = DeepFace.represent(face_crop_rgb, model_name="Facenet", enforce_detection=False)[0]['embedding']
                    face_embedding = np.array(face_embedding)

                    recognized_student = "Unknown"
                    min_distance = float("inf")

                    for student, encoding in student_encodings.items():
                        similarity = np.linalg.norm(encoding - face_embedding)
                        if similarity < min_distance and similarity < 10:
                            min_distance = similarity
                            recognized_student = student

                    cv2.putText(frame, recognized_student, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                except Exception as e:
                    print(f"⚠️ Error processing face: {e}")

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    print("✅ Real-time detection stopped.")  # Debugging log
    
# Load or compute student encodings
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        student_encodings = pickle.load(f)
    print("✅ Loaded existing student encodings.")
else:
    student_encodings = {}

def stream_video(video_path):
    """Streams processed video frames with detected faces."""
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = face_model(frame)  # Detect faces
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_crop = frame[y1:y2, x1:x2]

                if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
                    continue

                try:
                    face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    face_embedding = DeepFace.represent(face_crop_rgb, model_name="Facenet", enforce_detection=False)[0]['embedding']
                    face_embedding = np.array(face_embedding)

                    recognized_student = "Unknown"
                    min_distance = float("inf")

                    for student, encoding in student_encodings.items():
                        similarity = np.linalg.norm(encoding - face_embedding)
                        if similarity < min_distance and similarity < 10:
                            min_distance = similarity
                            recognized_student = student
                
                    cv2.putText(frame, recognized_student, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                except Exception as e:
                    print(f"⚠️ Error processing face: {e}")

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

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

@app.route("/detect")
def detect():

    return render_template("detect.html")
@app.route('/realtime_page')
def realtime_page():
    return render_template('realtime.html')
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
