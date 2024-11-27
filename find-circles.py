import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

# Real-world diameter of the M&M in meters
D = 0.0104 

# Print the camera Matrix
# b = np.load('//home/depei/Robotics/dynamixel_robot_arm/camera_vision/bin/camera_0_calibration.npz')
# print(b['camera_matrix']) #,b['dist'],b['rvecs'],b['tvecs'] 'camera_matrix', 'dist', 'rvecs', and 'tvecs'

# Load the camera calibration data
calibration_data = np.load('camera_vision/bin/camera_0_calibration.npz')
camera_matrix = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist']
rvecs = calibration_data['rvecs']
tvecs = calibration_data['tvecs']

img = cv2.imread('mnms.jpg')
# Changing the order from bgr to rgb so that matplotlib can show it
b,g,r = cv2.split(img)
img = cv2.merge([r,g,b])
plt.imshow(img)

# Convert the image from RGB to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

lower_thres = np.array([80,0,0])
upper_thres = np.array([255,120,100])

mask = cv2.inRange(img, lower_thres, upper_thres)

# Apply the mask to the original image
masked_frame = cv2.bitwise_and(hsv, hsv, mask=mask)

# Convert to grayscale
gray = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2GRAY)

# Apply Gaussian Blur
blurred = cv2.GaussianBlur(gray, (9, 9), 2)

min_radius = 20
max_radius = 50

circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1.5,                  # Increase dp for a better balance between accuracy and speed
    minDist=max_radius,      # Adjust based on m&m size and robots image resolution
    param1=100,              # Canny high threshold
    param2=20,               # Accumulator threshold, lower to detect more circles
    minRadius=min_radius,    # Set a smaller min radius
    maxRadius=max_radius     # Adjust max radius based on m&m size
)
img2 = img.copy()

centers = []

if circles is not None:
    circles = np.uint16(np.around(circles[0]))  # Access the first batch and round
    
    for circle in circles:
        x_c, y_c, r_image = circle  # Unpack x, y, and radius
        centers.append((x_c, y_c))
        
        # Draw the circle on the image
        center = (x_c, y_c)
        cv2.circle(img2, center, r_image, (255, 255, 0), 2)
        
        # Define the 3D point in real-world space
        # object_point = np.array([[0, 0, 0]], dtype=np.float32)  # Center of the M&M
        
        # Estimate depth (Z) using the apparent radius
        Z = (camera_matrix[0, 0] * D) / (2 * r_image)  # f_x is camera_matrix[0, 0]

        # Approximate a circle in 3D space for the M&M
        object_points_3D = []
        num_points = 10  # Number of points to approximate the circle
        radius_real_world = D / 2  # Real-world radius of the M&M
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            x = radius_real_world * np.cos(angle)
            y = radius_real_world * np.sin(angle)
            z = 0  # Circle is in the plane z = 0
            object_points_3D.append([x, y, z])
        object_points_3D = np.array(object_points_3D, dtype=np.float32)

        # Corresponding 2D image points (detected circle in the image)
        image_points_2D = []
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            x = x_c + r_image * np.cos(angle)
            y = y_c + r_image * np.sin(angle)
            image_points_2D.append([x, y])
        image_points_2D = np.array(image_points_2D, dtype=np.float32)

        
        # SolvePnP to find the translation vector
        _, rvec, tvec = cv2.solvePnP(
            object_points_3D,
            image_points_2D,
            camera_matrix,
            dist_coeffs,
            useExtrinsicGuess=False
        )
        
        # Translation vector
        translation_vector = tvec.flatten()
        print(f"Circle at ({x_c}, {y_c}) -> Translation Vector: {translation_vector}")
else:
    print("No circles found in the detected array.")
    
plt.imshow(img2)
plt.show()
# "centers" contains the center of each M&M. Now, convert this to real world coordinates 