import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Initialize variables
corner_points = []  # List to store corner points
corners_selected = 0
checkerboard_size = (3, 3)  # Define the checkerboard size
square_size = 10  # Square size in mm


# manualy enter corner points
# corner_points =  [(280, 308), (373, 322), (467, 338), (429, 414), (335, 399), (241, 380), (201, 456), (294, 476), (392, 499)]


# Load the checkerboard image
image_path = r"C:\\PATH_HERE\\"
name = "IMAGE_NAME_HERE"
full_path = image_path + name
image = cv2.imread(full_path)


# Mouse callback function to capture the click event
def select_corners(event, x, y, flags, param):
    global corners_selected
    if event == cv2.EVENT_LBUTTONDOWN:
        corner_points.append((x, y))
        corners_selected += 1
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Checkerboard', image)
        
        if corners_selected == np.prod(checkerboard_size):
            print("All corners selected")
            cv2.setMouseCallback('Checkerboard', lambda *args: None)
            cv2.destroyAllWindows()



# Resize the image
image = cv2.resize(image, (500, 500))

# Show the image and set up the mouse callback function
cv2.imshow('Checkerboard', image)
cv2.setMouseCallback('Checkerboard', select_corners)

print(f"checkerboard size: {checkerboard_size}")
print(f"Select {np.prod(checkerboard_size)} corners on the checkerboard.")
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert corner points to numpy array
corner_points = np.array(corner_points, dtype=np.float32)

# Convert BGR to RGB for matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(dpi=700)
plt.axis('off')
plt.imshow(image_rgb)
plt.scatter(corner_points[:, 0], corner_points[:, 1], color='red', s=50, marker='o')
plt.show()

# Define real-world 3D coordinates of the checkerboard corners in mm
obj_points = np.zeros((np.prod(checkerboard_size), 3), dtype=np.float32)
obj_points[:, :2] = np.indices(checkerboard_size).T.reshape(-1, 2) * square_size

# Prepare the image points (2D points from the selected corner points)
image_points = np.array(corner_points, dtype=np.float32).reshape(-1, 2)

# List to hold object points and image points for camera calibration
object_points = [obj_points]
image_points_list = [image_points]

# Perform camera calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    object_points, image_points_list, image.shape[1::-1], None, None)

# Output results
print("\nCamera matrix:")
print(camera_matrix)
print("\nDistortion coefficients:")
print(dist_coeffs)
print("\nRotation Vectors:")
print(rvecs)
print("\nTranslation Vectors:")
print(tvecs)

print("\nSelected corner points:")
print(corner_points)

# Define the file path for the CSV
csv_path = "C:\\PATH_HERE\\training.csv"

# Convert matrices to string for saving
camera_matrix_str = camera_matrix.flatten().tolist()
dist_coeffs_str = dist_coeffs.flatten().tolist()
rotation_matrix, _ = cv2.Rodrigues(rvecs[0])
rotation_matrix_str = rotation_matrix.flatten().tolist()
translation_vector_str = tvecs[0].flatten().tolist()

# Load existing CSV
df = pd.read_csv(csv_path)

index = 70
df.at[index, 'Camera_Matrix'] = str(camera_matrix_str)
df.at[index, 'Distortion_Coefficients'] = str(dist_coeffs_str)
df.at[index, 'Rotation_Matrix'] = str(rotation_matrix_str)
df.at[index, 'Translation_Vector'] = str(translation_vector_str)

# Save back to CSV
df.to_csv(csv_path, index=False)

print("\nCalibration results saved to CSV.")
