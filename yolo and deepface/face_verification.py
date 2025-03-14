
##verify same person or not
# from deepface import DeepFace  # Import DeepFace correctly

# # Paths to the images
# img1_path = "students/images/1/1.jpg"  # Replace with actual image path
# img2_path = "students/images/2/5.jpg"  # Replace with actual image path

# # Verify if both images belong to the same person
# result = DeepFace.verify(img1_path, img2_path, model_name="Facenet")

# # Print result
# if result["verified"]:
#     print("✅ Images are of the same person.")
# else:
#     print("❌ Images are of different people.")

## check student
from deepface import DeepFace
import os

# Path to stored student images
STUDENT_IMAGES_PATH = "students/students_images"

# Input image to verify
input_image_path = "students/images/7/2.jpg"  # Replace with the image you want to check

# Initialize the recognized student as "Unknown"
recognized_student = "Unknown"

# Iterate through student images
for student_image in os.listdir(STUDENT_IMAGES_PATH):
    student_image_path = os.path.join(STUDENT_IMAGES_PATH, student_image)

    try:
        # Compare input image with stored student image
        result = DeepFace.verify(input_image_path, student_image_path, model_name="Facenet")

        if result["verified"]:  # If match found
            recognized_student = os.path.splitext(student_image)[0]  # Get student name from filename
            break  # Stop checking once a match is found

    except Exception as e:
        print(f"Error processing {student_image_path}: {e}")

# Print recognized student's name
print(f"✅ Recognized Student: {recognized_student}")
