from dynamixel_sdk import * 
import cv2
import numpy as np
import time
import math

ADDR_MX_TORQUE_ENABLE = 24
ADDR_MX_CW_COMPLIANCE_MARGIN = 26
ADDR_MX_CCW_COMPLIANCE_MARGIN = 27
ADDR_MX_CW_COMPLIANCE_SLOPE = 28
ADDR_MX_CCW_COMPLIANCE_SLOPE = 29
ADDR_MX_GOAL_POSITION = 30
ADDR_MX_MOVING_SPEED = 32
ADDR_MX_PRESENT_POSITION = 36
ADDR_MX_PUNCH = 48
PROTOCOL_VERSION = 1.0
DXL_IDS = [1,2,3,4]
DEVICENAME = '/dev/tty.usbmodem1431401'
BAUDRATE = 1000000
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)
portHandler.openPort()
portHandler.setBaudRate(BAUDRATE)

for DXL_ID in DXL_IDS:
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)
    packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_CW_COMPLIANCE_MARGIN, 0)
    packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_CCW_COMPLIANCE_MARGIN, 0)
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_CW_COMPLIANCE_SLOPE, 32)
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_CCW_COMPLIANCE_SLOPE, 32)
    packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_MOVING_SPEED, 50)

def open_port():
    if not portHandler.openPort():
        print("Failed to open the port")
        quit()
    if not portHandler.setBaudRate(BAUDRATE):
        print("Failed to change the baudrate")
        quit()
    print("Port opened and baudrate set successfully")

def enable_torque():
    for DXL_ID in DXL_IDS:
        dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)
        if dxl_comm_result != COMM_SUCCESS:
            print(f"[ID:{DXL_ID}] %s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print(f"[ID:{DXL_ID}] %s" % packetHandler.getRxPacketError(dxl_error))
    print("Torque enabled for all motors")

def disable_torque():
    for DXL_ID in DXL_IDS:
        dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE)
        if dxl_comm_result != COMM_SUCCESS:
            print(f"[ID:{DXL_ID}] %s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print(f"[ID:{DXL_ID}] %s" % packetHandler.getRxPacketError(dxl_error))
    print("Torque disabled for all motors")

def close_port():
    portHandler.closePort()
    print("Port closed")

def angle_to_dynamixel_value(angle, min_angle=0, max_angle=300, resolution=1023):
    return int((angle - min_angle) * resolution / (max_angle - min_angle))

def move_to_position(positions):
    for i, DXL_ID in enumerate(DXL_IDS):
        dxl_goal_position = positions[i]
        dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_GOAL_POSITION, dxl_goal_position)
        if dxl_comm_result != COMM_SUCCESS:
            print(f"[ID:{DXL_ID}] %s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print(f"[ID:{DXL_ID}] %s" % packetHandler.getRxPacketError(dxl_error))
    time.sleep(5)  # Wait for 5 seconds to ensure the robot reaches the position

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
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Apply erosion to remove noise
    eroded_mask = cv2.erode(mask, kernel, iterations=1)

    # Apply dilation to fill gaps
    dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=2)

    # Apply the mask to the original image
    masked_frame = cv2.bitwise_and(img, img, mask=dilated_mask)

    if debug:
        cv2.imshow("masked_frame", masked_frame)

    # Convert to grayscale
    gray = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (7, 7), 2)

    if debug:
        cv2.imshow("blurred", blurred)

    min_radius = 30
    max_radius = 35

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
        if circles.shape[0] > 0:  # Ensure there is at least one circle detected
            circles = np.round(circles[0, :]).astype(np.int32)  # Accessing the first element
            for circle in circles:
                circles_found.append(circle)

    return circles_found

def draw_circles(circles, img):
    res = img.copy()
    if circles is not None:
        for circle in circles:
            cv2.circle(res, (circle[0], circle[1]), circle[2], (255, 255, 0), 2)
    return res

def get_real_world_circles(circles, camera_matrix, dist_coeffs, transformation_matrix):
    res = []
    if circles is not None:
        for circle in circles:
            circle = np.uint16(np.around(circle))  # Access the first batch and round
            x_c, y_c, r_image = circle  # Unpack x, y, and radius

            # Estimate depth (Z) using the apparent radius
            Z = (camera_matrix[0, 0] * D) / (2 * r_image)  # f_x is camera_matrix[0, 0]

            # Calculate the 3D coordinates of the circle center in the camera frame
            X = (x_c - camera_matrix[0, 2]) * Z / camera_matrix[0, 0]
            Y = (y_c - camera_matrix[1, 2]) * Z / camera_matrix[1, 1]

            # Transform the coordinates to the base frame of the robot
            point_camera_frame = np.array([X, Y, Z, 1])
            point_base_frame = np.dot(transformation_matrix, point_camera_frame)
            res.append(point_base_frame[:3])  # Convert back to 3D coordinates

    return res

def open_camera(cam_id=0):
    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FPS, 10)
    if not cap.isOpened():
        print(f"Failed to open camera with ID {cam_id}")
        return None
    return cap

def get_frame(device):
    ret, img = device.read()
    if not ret:
        print("Error capturing from video device.")
        return None
    return img

def cleanup(device):
    if device is not None:
        device.release()
    cv2.destroyAllWindows()

def calculate_joint_angles(x0, y0, z0, gamma):
    # Define the lengths of the links
    a1 = 50
    a2 = 93
    a3 = 93
    a4 = 50

    if x0 < 0:
        q1 = np.pi - np.arctan2(y0, x0) 
    elif x0 == 0 and y0 < 0:
        q1 = -np.pi/2
    elif x0 == 0 and y0 > 0:
        q1 = +np.pi/2
    else:
        q1 = np.arctan2(y0, x0)

    q3_in = ((x0 - a4 * np.cos(gamma))**2 + (z0 - a1 - a4 * np.sin(gamma))**2 - (a2**2 + a3**2)) / (2 * a2 * a3)
    q3_in = np.clip(q3_in, -1.0, 1.0)  # Clamp the value to the valid range for arccos
    q3_pos = np.arccos(q3_in)
    q3_neg = -np.arccos(q3_in)

    def calculate_q2_q4(q3):
        q2_in1 = (z0 - a1 - a4 * np.sin(gamma))
        q2_in2 = (x0 - a4 * np.cos(gamma))
        q2_in3 = a3 * np.sin(q3)
        q2_in4 = a2 + a3 * np.cos(q3)

        q2 = np.arctan2(q2_in1, q2_in2) - np.arctan2(q2_in3, q2_in4)
        q4 = gamma - q2 - q3
        return q2, q4

    q2_pos, q4_pos = calculate_q2_q4(q3_pos)
    q2_neg, q4_neg = calculate_q2_q4(q3_neg)

    return (q1, q2_pos, q3_pos, q4_pos), (q1, q2_neg, q3_neg, q4_neg)

def calculate_camera_pose(angles):
    # Define the lengths of the links
    link_lengths = [50, 93, 93, 50]

    # Calculate the positions of the joints using DH parameters
    x = [0]
    y = [0]
    z = [0]
    
    # Base joint (revolves around z-axis)
    theta1 = np.deg2rad(angles[0])
    d1 = link_lengths[0]
    x.append(x[-1])
    y.append(y[-1])
    z.append(z[-1] + d1)
    
    # First revolute joint
    theta2 = np.deg2rad(angles[1])
    a2 = link_lengths[1]
    x.append(x[-1] + a2 * np.cos(theta1) * np.cos(theta2))
    y.append(y[-1] + a2 * np.sin(theta1) * np.cos(theta2))
    z.append(z[-1] + a2 * np.sin(theta2))
    
    # Second revolute joint
    theta3 = np.deg2rad(angles[2] + angles[1])
    a3 = link_lengths[2]
    x.append(x[-1] + a3 * np.cos(theta1) * np.cos(theta3))
    y.append(y[-1] + a3 * np.sin(theta1) * np.cos(theta3))
    z.append(z[-1] + a3 * np.sin(theta3))
    
    # Third revolute joint
    theta4 = np.deg2rad(angles[3] + angles[2] + angles[1])
    a4 = link_lengths[3]
    x.append(x[-1] + a4 * np.cos(theta1) * np.cos(theta4))
    y.append(y[-1] + a4 * np.sin(theta1) * np.cos(theta4))
    z.append(z[-1] + a4 * np.sin(theta4))
    
    return x[-1], y[-1], z[-1]

def main():
    open_port()
    enable_torque()

    # Define the inspection position
    inspection_position = [503, 500, 220, 210]

    # Move to inspection position
    print("Moving to inspection position...")
    move_to_position(inspection_position)
    print("Inspecting...")

    # Wait for 5 seconds before checking the camera
    time.sleep(5)

    # Load the camera calibration data
    calibration_data = np.load('camera_vision/bin/camera_0_calibration.npz')
    camera_matrix = calibration_data['camera_matrix']
    dist_coeffs = calibration_data['dist']
    rvecs = calibration_data['rvecs']
    tvecs = calibration_data['tvecs']

    # Open the camera
    camera_id = 0  # Adjust as needed
    dev = open_camera(camera_id)
    if dev is None:
        print("No valid camera found.")
        disable_torque()
        close_port()
        quit()

    # Capture a single frame
    img_orig = get_frame(dev)
    if img_orig is not None:
        circles = get_circles(img_orig, debug=True)

        if circles is not None:
            print(f"Number of circles found: {len(circles)}")
            print("Coordinates of circles in the camera frame:")
            for circle in circles:
                print(f"Circle center: ({circle[0]}, {circle[1]}) with radius {circle[2]}")

            # Placeholder for the transformation matrix from the camera to the base coordinate frame of the robot
            transformation_matrix = np.eye(4)  # Identity matrix as a placeholder

            # Calculate the 3D pose of the camera with respect to the robot
            camera_pose = calculate_camera_pose([0, 0, 0, 0])  # Replace with actual joint angles if needed
            transformation_matrix[:3, 3] = camera_pose

            real_world_coordinates = get_real_world_circles(circles, camera_matrix, dist_coeffs, transformation_matrix)
            print("Real-world coordinates of detected circles:", real_world_coordinates)

            # Draw detected circles on the image
            results_image = draw_circles(circles, img_orig)
            cv2.imshow("Detected Circles", results_image)
            cv2.waitKey(5000)  # Wait for 5 seconds to close the window
            cv2.destroyWindow("Detected Circles")

            # Calculate joint angles for each detected point
            gamma = np.deg2rad(-90)  # Convert -90 degrees to radians
            for coord in real_world_coordinates:
                x0, y0, z0 = coord  # Extract the coordinates
                joint_angles_pos, joint_angles_neg = calculate_joint_angles(x0, y0, z0, gamma)
                print(f"Joint angles for position ({x0}, {y0}, {z0}):")
                print("Positive q3 solution:", joint_angles_pos)
                print("Negative q3 solution:", joint_angles_neg)

                # Check if the joint angles are valid (not NaN)
                if not any(np.isnan(joint_angles_pos)) and not any(np.isnan(joint_angles_neg)):
                    # Move to the positive q3 solution
                    move_to_position([angle_to_dynamixel_value(np.rad2deg(angle)) for angle in joint_angles_pos])
                    # Return to the inspection position
                    move_to_position(inspection_position)

                    # Move to the negative q3 solution
                    move_to_position([angle_to_dynamixel_value(np.rad2deg(angle)) for angle in joint_angles_neg])
                    # Return to the inspection position
                    move_to_position(inspection_position)
                else:
                    print("Invalid joint angles calculated, skipping this position.")
        else:
            print("No circles found in the detected array.")
    else:
        print("Failed to capture an image from the camera.")

    cleanup(dev)
    disable_torque()
    close_port()

if __name__ == "__main__":
    main()