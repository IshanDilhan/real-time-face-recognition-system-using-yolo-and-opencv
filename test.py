# with image

# from ultralytics import YOLO

# # Load the trained model
# model = YOLO("D:/Open cv real time face recognition system/model/best.pt")

# # Run inference on an image
# results = model("D:/Open cv real time face recognition system/Resouces/mytest/test4.jpeg", save=True, conf=0.5)

# # Show results
# for r in results:
#     r.show()


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
