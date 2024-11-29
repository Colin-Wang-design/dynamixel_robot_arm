from dynamixel_sdk import *
import numpy as np

def get_base_coordinate(translation_vector,DEVICENAME='/dev/tty.usbmodem1101', BAUDRATE=1000000):
    # Define constants
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
    DXL_IDS = [1, 2, 3, 4]
    ZERO_POSITIONS = [506, 211, 508, 826]
    TORQUE_ENABLE = 1
    TORQUE_DISABLE = 0

    # Validate translation vector
    if not (isinstance(translation_vector, (list, tuple)) and len(translation_vector) == 3):
        raise ValueError("Translation vector must be a list or tuple with 3 elements [x, y, z].")

    translation_vector = np.array(translation_vector)

    # Initialize PortHandler and PacketHandler
    portHandler = PortHandler(DEVICENAME)
    packetHandler = PacketHandler(PROTOCOL_VERSION)

    # Open port and set baudrate
    if not (portHandler.openPort() and portHandler.setBaudRate(BAUDRATE)):
        raise IOError("Failed to open port or set baudrate.")

    # Initialize servos
    try:
        for DXL_ID in DXL_IDS:
            packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)
            packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_CW_COMPLIANCE_MARGIN, 0)
            packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_CCW_COMPLIANCE_MARGIN, 0)
            packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_CW_COMPLIANCE_SLOPE, 32)
            packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_CCW_COMPLIANCE_SLOPE, 32)
            packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_MOVING_SPEED, 50)

        # Read angles
        angles_in_radians = []
        for i, DXL_ID in enumerate(DXL_IDS):
            dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler, DXL_ID, ADDR_MX_PRESENT_POSITION)
            if dxl_comm_result != COMM_SUCCESS:
                raise RuntimeError(packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                raise RuntimeError(packetHandler.getRxPacketError(dxl_error))
            else:
                zero_position = ZERO_POSITIONS[i]
                angle_rad = (dxl_present_position - zero_position) * 300.0 / 1023.0 * (np.pi / 180.0)
                angles_in_radians.append(angle_rad)

        # Define transformation matrix function
        def T_4_0(theta1, theta2, theta3, theta4):
            theta_sum = theta2 + theta3 + theta4
            r11 = np.cos(theta1) * np.cos(theta_sum)
            r21 = np.sin(theta1) * np.cos(theta_sum)
            r31 = np.sin(theta_sum)
            r12 = -np.sin(theta_sum) * np.cos(theta1)
            r22 = -np.sin(theta1) * np.sin(theta_sum)
            r32 = np.cos(theta_sum)
            r14 = np.cos(theta1) * (93 * np.cos(theta2) + 93 * np.cos(theta2 + theta3) + 85 * np.cos(theta_sum))
            r24 = np.sin(theta1) * (93 * np.cos(theta2) + 93 * np.cos(theta2 + theta3) + 85 * np.cos(theta_sum))
            r34 = 50 + 93 * np.sin(theta2) + 93 * np.sin(theta2 + theta3) + 85 * np.sin(theta_sum)
            return np.array([
                [r11, r12, np.sin(theta1), r14],
                [r21, r22, -np.cos(theta1), r24],
                [r31, r32, 0, r34],
                [0, 0, 0, 1]
            ])

        # Camera matrix transformation
        T_5_4 = np.array([
            [1, 0, 0, -45],
            [0, 1, 0, 38],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        theta1, theta2, theta3, theta4 = angles_in_radians
        T_4_0_matrix = T_4_0(theta1, theta2, theta3, theta4)
        T_5_0 = np.dot(T_4_0_matrix, T_5_4)

        # # Extract camera matrix's last column (first three components)
        # camera_position = T_5_0[:3, 3]
        # camera_position[0] -= 10
        
        # Define the object's position in the camera frame
        T_object_to_camera = np.array(translation_vector)
        # Add homogeneous coordinate to the object vector
        T_object_to_camera_homogeneous = np.append(T_object_to_camera, 1)  # Convert to [x, y, z, 1]

        # Calculate the object's position in the base frame
        T_object_to_base_homogeneous = np.dot(T_5_0, T_object_to_camera_homogeneous)

        # Extract the translation part from the resulting homogeneous vector
        T_object_to_base = T_object_to_base_homogeneous[:3].tolist()
        # Round to three decimal places
        T_object_to_base = [round(coord, 3) for coord in T_object_to_base]
        return T_object_to_base

    finally:
        # Disable torque and close port
        for DXL_ID in DXL_IDS:
            packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE)
        portHandler.closePort()

if __name__ == "__main__":
    vector=[0.0,0.0,0.0]
    translation_vector = get_base_coordinate(translation_vector=vector)
    print("Camera translation:",translation_vector)