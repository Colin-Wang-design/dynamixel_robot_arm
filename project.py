from dynamixel_sdk import * 
import numpy as np

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
ZERO_POSITIONS = [506, 202, 522, 826]  # Zero positions for each servo of robot 6
DEVICENAME = '/dev/ttyACM0' # /dev/tty.usbmodem1301
BAUDRATE = 1000000
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Open port and set baudrate
if portHandler.openPort() and portHandler.setBaudRate(BAUDRATE):
    print("Port opened and baudrate set.")
else:
    print("Failed to open port or set baudrate.")
    quit()
    
for DXL_ID in DXL_IDS:
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)
    packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_CW_COMPLIANCE_MARGIN, 0)
    packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_CCW_COMPLIANCE_MARGIN, 0)
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_CW_COMPLIANCE_SLOPE, 32)
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_CCW_COMPLIANCE_SLOPE, 32)
    packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_MOVING_SPEED, 50)
    # packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_MX_GOAL_POSITION, 512)
    
# Read and compute angles in radians
angles_in_radians = []
for i, DXL_ID in enumerate(DXL_IDS):
    dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler, DXL_ID, ADDR_MX_PRESENT_POSITION)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % packetHandler.getRxPacketError(dxl_error))
    else:
        # Compute angle in radians
        zero_position = ZERO_POSITIONS[i]
        angle_rad = (dxl_present_position - zero_position) * 300.0 / 1023.0 * (3.141592653589793 / 180.0)
        angles_in_radians.append(angle_rad)
        print("[ID:%03d] PresPos:%03d -> Angle (rad): %.4f" % (DXL_ID, dxl_present_position, angle_rad))

    # print("[ID:%03d]  PresPos:%03d" % (DXL_ID, dxl_present_position))

for DXL_ID in DXL_IDS:
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE)
    
# Define T_4^0 as a function of given parameters
def T_4_0(theta1, theta2, theta3, theta4):
    theta_sum = theta2 + theta3 + theta4
    
    r11 = np.cos(theta1) * np.cos(theta_sum)
    r21 = np.sin(theta1) * np.cos(theta_sum)
    r31 = np.sin(theta_sum)
    
    r12 = -np.sin(theta_sum) * np.cos(theta1)
    r22 = -np.sin(theta1) * np.sin(theta_sum)
    r32 = np.cos(theta_sum)
    
    r14 = np.cos(theta1) * (93 * np.cos(theta2) + 93 * np.cos(theta2 + theta3) + 50 * np.cos(theta_sum))
    r24 = np.sin(theta1) * (93 * np.cos(theta2) + 93 * np.cos(theta2 + theta3) + 50 * np.cos(theta_sum))
    r34 = 50 + 93 * np.sin(theta2) + 93 * np.sin(theta2 + theta3) + 50 * np.sin(theta_sum)
    
    return np.array([
        [r11, r12, np.sin(theta1), r14],
        [r21, r22, -np.cos(theta1), r24],
        [r31, r32, 0, r34],
        [0, 0, 0, 1]
    ])

# Define T_5^4
T_5_4 = np.array([
    [1, 0, 0, -15],
    [0, 1, 0, 45],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

# Example angles in radians
theta1 = angles_in_radians[0]  # Replace with actual values if known
theta2 = angles_in_radians[1]
theta3 = angles_in_radians[2]
theta4 = angles_in_radians[3]

# Compute T_4^0
T_4_0_matrix = T_4_0(theta1, theta2, theta3, theta4)

# Compute T_5^0
T_5_0 = np.dot(T_4_0_matrix, T_5_4)

# Print the result
print("T_4^0 Matrix:")
for row in T_4_0_matrix:
    print(" ".join(f"{value:10.4f}" for value in row))

# Close port
portHandler.closePort()
