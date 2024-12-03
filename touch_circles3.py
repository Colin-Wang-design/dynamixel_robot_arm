from dynamixel_sdk import * 
import cv2
import numpy as np
import time
import math
from robot_sim.IK_2 import calculate_joint_angles


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
DEVICENAME = '/dev/tty.usbmodem14324401'
BAUDRATE = 1000000
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)
portHandler.openPort()
portHandler.setBaudRate(BAUDRATE)
# Define the zero positions for each joint
zero_positions = [503, 210, 506, 515]





def T05_matrix(q1, q2, q3, q4):
    # Define the transformation matrices
    T_1_0 = np.array([
        [np.cos(q1), 0, np.sin(q1), 0],
        [np.sin(q1), 0, -np.cos(q1), 0],
        [0, 1, 0, 50],
        [0, 0, 0, 1]
    ])

    T_2_1 = np.array([
        [np.cos(q2), -np.sin(q2), 0, 93 * np.cos(q2)],
        [np.sin(q2), np.cos(q2), 0, 93 * np.sin(q2)],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    T_3_2 = np.array([
        [np.cos(q3), -np.sin(q3), 0, 93 * np.cos(q3)],
        [np.sin(q3), np.cos(q3), 0, 93 * np.sin(q3)],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    T_4_3 = np.array([
        [np.cos(q4), -np.sin(q4), 0, 50 * np.cos(q4)],
        [np.sin(q4), np.cos(q4), 0, 50 * np.sin(q4)],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    T_5_4 = np.array([
        [1, 0, 0, -15],
        [0, 1, 0, 45],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Calculate the overall transformation matrix
    T_5_0 = T_1_0 @ T_2_1 @ T_3_2 @ T_4_3 @ T_5_4
    return T_5_0

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

def cam_2_end_effector(x, y, z, gamma, q1):
    # Camera to end effector transformation
    x2x = 45
    z2z = 15

    x_dist = (x2x*np.sin(gamma) + z2z*np.cos(gamma))*np.cos(q1)
    y_dist = (x2x*np.sin(gamma) + z2z*np.cos(gamma))*np.sin(q1)
    
    z_dist = - x2x*np.cos(gamma) + z2z*np.sin(gamma)
    return x+x_dist, y+y_dist, z+z_dist

def move_to_position(positions):
    for i, DXL_ID in enumerate(DXL_IDS):
        dxl_goal_position = positions[i]
        dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_GOAL_POSITION, dxl_goal_position)
        if dxl_comm_result != COMM_SUCCESS:
            print(f"[ID:{DXL_ID}] %s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print(f"[ID:{DXL_ID}] %s" % packetHandler.getRxPacketError(dxl_error))
    time.sleep(2)  # Wait for 5 seconds to ensure the robot reaches the position

def angle_to_dynamixel_value(angle, min_angle=0, max_angle=300, resolution=1023):
    return int((angle - min_angle) * resolution / (max_angle - min_angle))

def offset_position(positions, zero_positions):
    return [pos+offset for pos, offset in zip(positions, zero_positions)]

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



def draw_circles(circles, image):
    """
    Draw circles and their numbers on the image.

    :param circles: List of circles with their 2D coordinates and radius [(x, y, r), ...]
    :param image: The image on which to draw the circles
    :return: The image with circles and numbers drawn
    """
    for i, circle in enumerate(circles):
        center = (circle[0], circle[1])
        radius = circle[2]
        # Draw the circle
        cv2.circle(image, center, radius, (255, 255, 0), 2)
        # Draw the circle number
        cv2.putText(image, str(i+1), (center[0] - 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def calculate_3d_positions(circles, camera_matrix, depth):
    """
    Calculate the 3D positions of the circles in the camera's coordinate system.

    :param circles: List of circles with their 2D coordinates and radius [(x, y, r), ...]
    :param camera_matrix: Camera intrinsic matrix
    :param depth: Depth of the circles from the camera lens in mm
    :return: List of 3D coordinates [(X, Y, Z), ...]
    """
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    positions_3d = []
    for circle in circles:
        x, y, _ = circle
        # X = (x - cx) * depth / fx
        # Y = (y - cy) * depth / fy
        # Z = depth

        Z = (x - cx) * depth / fx
        Y = -(y - cy) * depth / fy-20
        X = depth

        positions_3d.append((X, Y, Z))

    return positions_3d


def calculate_3d_positions_pnp(circles, camera_matrix, dist_coeffs, depth):
    """
    Calculate the 3D positions of the circles in the camera's coordinate system using PnP.

    :param circles: List of circles with their 2D coordinates and radius [(x, y, r), ...]
    :param camera_matrix: Camera intrinsic matrix
    :param dist_coeffs: Camera distortion coefficients
    :param depth: Depth of the circles from the camera lens in mm
    :return: List of 3D coordinates [(X, Y, Z), ...]
    """
    # 2D image points
    image_points = np.array([(circle[0], circle[1]) for circle in circles], dtype=np.float32)

    # 3D world points (assuming depth is the X coordinate)
    world_points = np.array([(depth, (circle[0] - camera_matrix[0, 2]) * depth / camera_matrix[0, 0], 
                              (circle[1] - camera_matrix[1, 2]) * depth / camera_matrix[1, 1]) for circle in circles], dtype=np.float32)

    # Solve PnP to get the rotation and translation vectors
    success, rotation_vector, translation_vector = cv2.solvePnP(world_points, image_points, camera_matrix, dist_coeffs)

    if not success:
        raise ValueError("PnP solution could not be found")

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Combine rotation matrix and translation vector into a transformation matrix
    transformation_matrix = np.hstack((rotation_matrix, translation_vector))
    transformation_matrix = np.vstack((transformation_matrix, [0, 0, 0, 1]))

    # Transform the 3D world points to the camera coordinate system
    positions_3d = []
    for point in world_points:
        point_homogeneous = np.append(point, 1).reshape(4, 1)
        point_camera_frame = transformation_matrix @ point_homogeneous
        positions_3d.append(point_camera_frame[:3].flatten())

    return positions_3d

def main():
    open_port()
    enable_torque()
    # Inspection position coordinates [x, y, z, gamma] (in mm and degrees) - camera coordinates
    # Inspection position 1
    
    x, y, z, gamma = 150, 0, 100, -90 #141.63254771866687, -5.19737006392425, -19.999999999999986, -90
    depth = z + 19  # Depth of the circles from the camera lens in mm
    print("Desired Camera position 1: ", x, y, z, gamma)
    # Transformation from camera to end effector
    x, y, z = cam_2_end_effector(x, y, z, np.deg2rad(gamma), 0)
    print("Desired End effector position 1: ", x, y, z)
    # Inverse kinematics for end effector position 1 (based on camera position 1)
    q1, q2, q3, q4 = calculate_joint_angles(x, y, z, np.deg2rad(gamma))[1] # elbow up
    print("Joint angles 1: ", np.degrees(q1), np.degrees(q2), np.degrees(q3), np.degrees(q4))
    qs = [q1, q2, q3, q4]
    goal_positions = [zero_positions[i] + angle_to_dynamixel_value(np.degrees(qs[i])) for i in range(4)]
    goal_inspection_position_1 = goal_positions
    print("Goal positions 1: ", goal_positions)

    # Move to inspection position 1
    print("Moving to inspection position...")
    move_to_position(goal_inspection_position_1)
    print("Inspecting...")

    # Wait for 5 seconds before checking the camera
    time.sleep(1)

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
            # Draw detected circles on the image
            results_image = draw_circles(circles, img_orig)
            cv2.imshow("Detected Circles", results_image)
            cv2.waitKey(2000)  # Wait for 5 seconds to close the window
            cv2.destroyWindow("Detected Circles")

            # Calculate the 3D positions of the circles
            positions_3d = calculate_3d_positions(circles, camera_matrix, depth)
            print("positions_3d: ", positions_3d)
            print("3D positions of the circles:")
            for i, pos in enumerate(positions_3d):
                print(f"Circle {i+1}: {pos}")

            # Transform the 3D positions of the circles in the camera frame to the robot frame
            # (camera to end effector transformation)
            print("qs: ", qs)
            T_5_0 = T05_matrix(qs[0], qs[1], qs[2], qs[3])
            print("T_5_0: ", T_5_0)
            positions_3d_robot = []
            for pos in positions_3d:
                # Convert the position to homogeneous coordinates
                pos_homogeneous = np.array([pos[0], pos[1], pos[2], 1]).reshape(4, 1)
                print("pos_homogeneous: ", pos_homogeneous)
                
                # Transform the position from the camera frame to the robot base frame
                pos_robot_frame =  T_5_0 @ pos_homogeneous
                # Convert np.float64 to standard Python float
                pos_robot_frame_float = [float(coord) for coord in pos_robot_frame[:3].flatten()]
                positions_3d_robot.append(pos_robot_frame_float)

            print("3D positions of the circles in the robot base frame:")
            gamma = np.deg2rad(gamma)
            for i, pos in enumerate(positions_3d_robot):
                print(f"Circle {i+1} in the robot frame: {pos}")
                # Calculate the inverse kinematics for the end effector position
                joint_angles = calculate_joint_angles(pos[0], pos[1], pos[2], gamma)[1] # elbow up
                print(f"Joint angles for circle {i+1}: {joint_angles}")
                
                # Check if the joint angles are valid (not NaN)
                if not any(np.isnan(joint_angles)):
                    qs = joint_angles
                    goal_positions = [zero_positions[i] + angle_to_dynamixel_value(np.degrees(qs[i])) for i in range(4)]
                    print(f"Circle {i+1} goal position: ", goal_positions)
                    move_to_position(goal_positions)
                    print(f"Touching circle {i+1}...")
                    time.sleep(2)
                    # Return to inspection position
                    move_to_position(goal_inspection_position_1)
                    print("Returning to inspection position...")
                else:
                    print(f"Invalid joint angles calculated for circle {i+1}, skipping this position.")
            
        else:
            print("No circles found in the detected array.")
    else:
        print("Failed to capture an image from the camera.")

    cleanup(dev)
    close_port()


if __name__ == "__main__":
    main()