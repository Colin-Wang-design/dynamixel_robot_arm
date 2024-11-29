#!/usr/bin/python
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from camera_transform import get_base_coordinate

# Real-world diameter of the M&M in meters
D = 0.014 

# Print the camera Matrix
# b = np.load('//home/depei/Robotics/dynamixel_robot_arm/camera_vision/bin/camera_0_calibration.npz')
# print(b['camera_matrix']) #,b['dist'],b['rvecs'],b['tvecs'] 'camera_matrix', 'dist', 'rvecs', and 'tvecs'

# Load the camera calibration data
calibration_data = np.load('camera_vision/bin/camera_0_calibration.npz')
camera_matrix = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist']
rvecs = calibration_data['rvecs']
tvecs = calibration_data['tvecs']


def get_circles(img, debug=False):

    # Convert the image from RGB to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Define HSV thresholds for red (two ranges)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([11, 255, 255])
    lower_red2 = np.array([110, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for both red ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine masks
    mask = mask1 + mask2

    kernel = np.ones((5, 5), np.uint8)  # Define the kernel size; adjust as needed


    # Apply erosion to remove noise
    eroded_mask = cv2.erode(mask, kernel, iterations=1)

    # Apply dilation to fill gaps
    dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=2)



    # Apply the mask to the original image
    masked_frame = cv2.bitwise_and(img, img, mask=dilated_mask)
    #masked_frame = cv2.bitwise_and(img, img, mask=mask)

    if debug:
        cv2.imshow("masked_frame", masked_frame)
    

    # Convert to grayscale
    gray = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (7,7), 2)

    if debug:
        cv2.imshow("blurred", blurred)

    min_radius = 35
    max_radius = 46

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

    circles_found = []

    if circles is not None:
        #print(f"Type of circles: {type(circles)}")
        # Check the shape of circles
        #print(f"Shape of circles: {circles.shape}")  # Expecting shape (1, n, 3)
        
        if circles.shape[0] > 0:  # Ensure there is at least one circle detected
            circles = np.round(circles[0, :]).astype(np.int32)  # Accessing the first element
            
            for circle in circles:
                circles_found.append(circle)

                # Draw the circle on the image
                #cv2.circle(img2, center, r, (255, 255, 0), 2)
    
    return circles_found

def draw_circles(circles, img):
    res = img.copy()
    if circles is not None:
        for circle in circles:
            cv2.circle(res, (circle[0], circle[1]), circle[2], (255, 255, 0), 2)
    return res

def get_real_world_circles(circles):
    res = []
    if circles is not None:
        
        for circle in circles:
            circle = np.uint16(np.around(circle))  # Access the first batch and round
            x_c, y_c, r_image = circle  # Unpack x, y, and radius
            
            # Estimate depth (Z) using the apparent radius
            #Z = (camera_matrix[0, 0] * D) / (2 * r_image)  # f_x is camera_matrix[0, 0]

            # Approximate a circle in 3D space for the M&M
            num_points = 10  # Number of points to approximate the circle
            radius_real_world = D / 2  # Real-world radius of the M&M
            object_points_3D = []
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
            
            # Flatten tvec to get x, y, z as a simple list
            tvec_flat = 1000*tvec.flatten()
            tvec_swapped = [tvec_flat[2], tvec_flat[1], tvec_flat[0]]  # Swap z (index 2) with x (index 0)
            # Round to three decimal places
            tvec_swapped = [round(coord, 3) for coord in tvec_swapped]

            res.append((object_points_3D, image_points_2D, rvec, tvec_swapped))
    return res

 

##
# Opens a video capture device with a resolution of 800x600 at 30 FPS.
##
def open_camera(cam_id = 0):
    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FPS, 10)
    return cap
 
##
# Gets a frame from an open video device, or returns None
# if the capture could not be made.
##
def get_frame(device):
    ret, img = device.read()
    if (ret == False): # failed to capture
        print("Error capturing from video device.")
        return None
    return img
 
##
# Closes all OpenCV windows and releases video capture device before exit.
##
def cleanup(cam_id = 0): 
    cv2.destroyAllWindows()
    cv2.VideoCapture(cam_id).release()
 

if __name__ == "__main__":
    # Camera ID to read video from (numbered from 0)
    camera_id = 0 # Linux = 0; Mac = 2
    dev = open_camera(camera_id) # open the camera as a video capture device
 
    while True:
        img_orig = get_frame(dev) # Get a frame from the camera
        if img_orig is not None: # if we did get an image
            cv2.imshow("video", img_orig) # display the image in a window named "video"
            circles = get_circles(img_orig, debug=True)

            results_image = img_orig.copy()

            if circles is not None:
                #circles = np.uint16(np.around(circles[0]))  # Access the first batch and round

                results_image = draw_circles(circles, results_image)

                obj = get_real_world_circles(circles)
                if obj is not None:
                    for o in obj:
                        # Object to base frame trans-vector
                        translation_vector = get_base_coordinate(translation_vector=o[3])
                        print(f"Circle: ({o[1][0][0]}, {o[1][0][1]}) -> T-Vector: {translation_vector}")#    
                    print(" ")# Gap between every loop                         
                else:
                    print("failed to get real world circles")
            else:
                print("No circles found in the detected array.")
                            
            # Display Results
            cv2.imshow("results", results_image)
        else: # if we failed to capture (camera disconnected?), then quit
            break
 
        if (cv2.waitKey(2) >= 0): # If the user presses any key, exit the loop
            break
        
 
    cleanup(camera_id) # close video device and windows before we exit