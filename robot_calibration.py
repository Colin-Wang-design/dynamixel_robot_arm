from dynamixel_sdk import * 
import time
import math

ADDR_MX_TORQUE_ENABLE = 24
ADDR_MX_PRESENT_POSITION = 36
PROTOCOL_VERSION = 1.0
DXL_IDS = [1, 2, 3, 4]
DEVICENAME = '/dev/tty.usbmodem14224101'
BAUDRATE = 1000000
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

# Disable torque for all motors
for DXL_ID in DXL_IDS:
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE)
    if dxl_comm_result != COMM_SUCCESS:
        print(f"[ID:{DXL_ID}] %s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print(f"[ID:{DXL_ID}] %s" % packetHandler.getRxPacketError(dxl_error))

# Function to convert Dynamixel values to angles
def dynamixel_value_to_angle(value, min_angle=0, max_angle=300, resolution=1023):
    return (value * (max_angle - min_angle) / resolution) + min_angle

try:
    while True:
        for DXL_ID in DXL_IDS:
            dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler, DXL_ID, ADDR_MX_PRESENT_POSITION)
            if dxl_comm_result != COMM_SUCCESS:
                print(f"[ID:{DXL_ID}] %s" % packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print(f"[ID:{DXL_ID}] %s" % packetHandler.getRxPacketError(dxl_error))
            else:
                angle = dynamixel_value_to_angle(dxl_present_position)
                print(f"[ID:{DXL_ID:03d}] Angle: {angle:.2f} degrees")

        time.sleep(1)  # Adjust the sleep time as needed

except KeyboardInterrupt:
    print("Terminating the script...")

finally:
    # Close port
    portHandler.closePort()