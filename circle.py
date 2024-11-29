from dynamixel_sdk import * 
import openpyxl
import time
import math

ADDR_MX_TORQUE_ENABLE = 24
ADDR_MX_GOAL_POSITION = 30
ADDR_MX_PRESENT_POSITION = 36
PROTOCOL_VERSION = 1.0
DXL_IDS = [1, 2, 3, 4]
DEVICENAME = '/dev/tty.usbmodem1423101'
BAUDRATE = 1000000
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Open port
if not portHandler.openPort():
    print("Failed to open the port")
    quit()

# Set port baudrate
if not portHandler.setBaudRate(BAUDRATE):
    print("Failed to change the baudrate")
    quit()

# Enable torque for all motors
for DXL_ID in DXL_IDS:
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)
    if dxl_comm_result != COMM_SUCCESS:
        print(f"[ID:{DXL_ID}] %s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print(f"[ID:{DXL_ID}] %s" % packetHandler.getRxPacketError(dxl_error))

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
    time.sleep(5)  # Wait for 5 seconds to ensure the robot reaches the zero configuration

# Load the sequence of moves from an Excel file
wb = openpyxl.load_workbook('data/test.xlsx')  # _neg is elbow up configurations
sheet = wb.active

sequence_of_moves = []

# Read every second row starting from the second row, and columns 3 to 6
for i, row in enumerate(sheet.iter_rows(min_row=2, min_col=3, max_col=6, values_only=True)):
    # Convert radians to degrees
    sequence_of_moves.append([math.degrees(value) for value in row])

# Print the shape of the sequence of moves
print("Shape of the sequence of moves:", len(sequence_of_moves), len(sequence_of_moves[0]))

# Define the zero positions for each joint
zero_positions = [503, 210, 506, 530]  # robot 6

# Check zero configuration
check_zero_config(zero_positions)

# Print the positions it will go through
for index, goal_positions in enumerate(sequence_of_moves):
    displaced_positions = []
    for i, goal_position in enumerate(goal_positions):
        displaced_position = zero_positions[i] + int(angle_to_dynamixel_value(goal_position))
        displaced_positions.append(displaced_position)
    print(f"Config {index + 1}: {displaced_positions}")

# Perform the sequence of moves
for index, goal_positions in enumerate(sequence_of_moves):
    displaced_positions = []
    for i, goal_position in enumerate(goal_positions):
        displaced_position = zero_positions[i] + int(angle_to_dynamixel_value(goal_position))
        displaced_positions.append(displaced_position)
        dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_IDS[i], ADDR_MX_GOAL_POSITION, displaced_position)
        if dxl_comm_result != COMM_SUCCESS:
            print(f"[ID:{DXL_IDS[i]}] %s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print(f"[ID:{DXL_IDS[i]}] %s" % packetHandler.getRxPacketError(dxl_error))
    print(f"Moving to Config {index + 1}: {displaced_positions}")
    time.sleep(0.3)  # Adjust the sleep time as needed

# Disable torque for all motors
for DXL_ID in DXL_IDS:
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)
    if dxl_comm_result != COMM_SUCCESS:
        print(f"[ID:{DXL_ID}] %s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print(f"[ID:{DXL_ID}] %s" % packetHandler.getRxPacketError(dxl_error))

# Close port
portHandler.closePort()