from dynamixel_sdk import * 
import openpyxl
import time
import math
import numpy as np
from robot_sim.IK_2 import calculate_joint_angles  # Import the inverse kinematics function

# Constants
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

# Open port and set baudrate
if not portHandler.openPort():
    print("Failed to open the port")
    quit()
if not portHandler.setBaudRate(BAUDRATE):
    print("Failed to change the baudrate")
    quit()
print("Port opened and baudrate set successfully")

# Enable torque for all motors
for DXL_ID in DXL_IDS:
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)
    if dxl_comm_result != COMM_SUCCESS:
        print(f"[ID:{DXL_ID}] %s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print(f"[ID:{DXL_ID}] %s" % packetHandler.getRxPacketError(dxl_error))
print("Torque enabled for all motors")

# Function to convert angles to Dynamixel values
def angle_to_dynamixel_value(angle, min_angle=0, max_angle=300, resolution=1023):
    return int((angle - min_angle) * resolution / (max_angle - min_angle))

# Function to check zero configuration
def check_zero_config(zero_positions):
    print("Moving to zero configuration...")
    for i, DXL_ID in enumerate(DXL_IDS):
        dxl_goal_position = zero_positions[i]
        dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_GOAL_POSITION, dxl_goal_position)
        if dxl_comm_result != COMM_SUCCESS:
            print(f"[ID:{DXL_ID}] %s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print(f"[ID:{DXL_ID}] %s" % packetHandler.getRxPacketError(dxl_error))
    time.sleep(8)  # Wait for 8 seconds to ensure the robot reaches the zero configuration

# Define the zero positions for each joint
zero_positions = [503, 210, 506, 500]  # robot 6

# Check zero configuration
check_zero_config(zero_positions)

# Placeholder for the desired inspection position coordinates and angle
inspection_position = [150, 0, 200]  # Replace with actual coordinates
gamma_deg = -90  # Replace with actual angle in degrees
gamma = np.deg2rad(gamma_deg)

# Calculate the joint angles using inverse kinematics
results_pos, results_neg = calculate_joint_angles(inspection_position[0], inspection_position[1], inspection_position[2], gamma)
print("results_pos: ", results_pos)
print("results_neg: ", results_neg)

# Choose one of the solutions (positive q3 in this case)
joint_angles = results_pos

# Convert joint angles to Dynamixel values and add to zero positions
goal_positions = [zero_positions[i] + angle_to_dynamixel_value(np.degrees(joint_angles[i])) for i in range(4)]
print(f"Moving to inspection position: {goal_positions}")

# Move to the inspection position
for i, DXL_ID in enumerate(DXL_IDS):
    dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_GOAL_POSITION, goal_positions[i])
    if dxl_comm_result != COMM_SUCCESS:
        print(f"[ID:{DXL_ID}] %s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print(f"[ID:{DXL_ID}] %s" % packetHandler.getRxPacketError(dxl_error))
time.sleep(5)  # Wait for 5 seconds to ensure the robot reaches the inspection position

# Disable torque for all motors
for DXL_ID in DXL_IDS:
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE)
    if dxl_comm_result != COMM_SUCCESS:
        print(f"[ID:{DXL_ID}] %s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print(f"[ID:{DXL_ID}] %s" % packetHandler.getRxPacketError(dxl_error))
print("Torque disabled for all motors")

# Close port
portHandler.closePort()
print("Port closed")