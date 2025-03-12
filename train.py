from ultralytics import YOLO
import torch
print("CUDA Available:", torch.cuda.is_available())
print("GPU Count:", torch.cuda.device_count())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")


# Load the pre-trained YOLO model (small version for fast training)
model = YOLO('yolov8n.pt')  # You can also try 'yolov8s.pt' or 'yolov8m.pt'

# Train the model
model.train(data='Resources/data.yaml', epochs=50, batch=16, imgsz=640, device='cuda')

