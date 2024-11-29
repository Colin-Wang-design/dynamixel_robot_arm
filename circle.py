from dynamixel_sdk import * 
import time
import openpyxl
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
DEVICENAME = '/dev/tty.usbmodem1101'
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

# Function to convert angles to Dynamixel values
def angle_to_dynamixel_value(angle, min_angle=0, max_angle=300, resolution=1023):
    return int((angle - min_angle) * resolution / (max_angle - min_angle))

# Load the sequence of moves from an Excel file
wb = openpyxl.load_workbook('joint_angles.xlsx') # _neg is elbow up configurations
sheet = wb.active

sequence_of_moves = []

# Read every second row starting from the second row, and columns 3 to 6
for i, row in enumerate(sheet.iter_rows(min_row=2, min_col=2, max_col=5, values_only=True)):
    # Convert radians to degrees
    sequence_of_moves.append([math.degrees(value) for value in row])

#print the shape of the sequence of moves
print("Shape of the sequence of moves:", len(sequence_of_moves), len(sequence_of_moves[0]))

# Define the zero positions for each joint
#zero_positions = [614, 205, 506, 810] #robot 9
zero_positions = [497, 203, 513, 815] #robot 6

# Print the positions it will go through
for index, goal_positions in enumerate(sequence_of_moves):
    displaced_positions = []
    for i, goal_position in enumerate(goal_positions):
        displaced_position = zero_positions[i] + int(angle_to_dynamixel_value(goal_position))
        displaced_positions.append(displaced_position)
    print(f"Config {index + 1}: {displaced_positions}")

# Execute the sequence of moves
for index, goal_positions in enumerate(sequence_of_moves):
    print(f"Processing point index: {index}")  # Print the index of the point being processed
    print("Original goal positions:", goal_positions)  # Print the original goal positions
    displaced_positions = []
    for i, goal_position in enumerate(goal_positions):
        print(" int Goal position:", int(angle_to_dynamixel_value(goal_position)))
        # Convert to degrees and displace
        displaced_position = zero_positions[i] + int(angle_to_dynamixel_value(goal_position))
        displaced_positions.append(displaced_position)
    print("Displaced goal positions:", displaced_positions)  # Print the displaced goal positions

    for DXL_ID, goal_position in zip(DXL_IDS, displaced_positions):
        dxl_comm_result, dxl_error = packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_GOAL_POSITION, goal_position)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % packetHandler.getRxPacketError(dxl_error))

    # Allow time for the motors to reach the goal positions
    #time.sleep(0.01)
        # Allow time for the motors to reach the goal positions
    if index == 1:
        time.sleep(5)
    else:
        time.sleep(0.1)

    # Read present position
    for DXL_ID in DXL_IDS:
        dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler, DXL_ID, ADDR_MX_PRESENT_POSITION)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % packetHandler.getRxPacketError(dxl_error))

        print("[ID:%03d]  PresPos:%03d" % (DXL_ID, dxl_present_position))

# Lock the position by enabling torque again
for DXL_ID in DXL_IDS:
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE)

portHandler.closePort()