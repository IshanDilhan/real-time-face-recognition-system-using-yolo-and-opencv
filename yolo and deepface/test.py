import os
import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("D:/Open cv real time face recognition system/model/best.pt")

# Define input & output folders
input_folder = r"D:\Open cv real time face recognition system\Resources\mytest"
output_folder = r"D:\Open cv real time face recognition system\Resources\results"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Debugging - Check if input folder exists
assert os.path.exists(input_folder), f"⚠️ Folder does not exist: {input_folder}"

# Loop through all images in folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(input_folder, filename)
        
        # Run inference
        results = model(image_path, conf=0.5)

        for r in results:
            # Get the image with bounding boxes
            img_with_boxes = r.plot()

            # Remove text by manually drawing boxes
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box.tolist())  # Get bounding box coordinates
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

            # Save the image
            output_path = os.path.join(output_folder, f"result_{filename}")
            cv2.imwrite(output_path, img_with_boxes)
            print(f"✅ Processed & saved: {output_path}")

print("🎯 All images processed successfully!")






#with frontcam
# import cv2
# from ultralytics import YOLO

# # Load your trained model
# model = YOLO("D:/Open cv real time face recognition system/model/best.pt")  # Update path if needed

# # Open webcam (0 = default camera)
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Run YOLO inference on the frame
#     results = model(frame, conf=0.5)

#     # Display results
#     annotated_frame = results[0].plot()  # Draw bounding boxes
#     cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()


