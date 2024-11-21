import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IK_2 import calculate_joint_angles  # Assuming the IK script is named IK_2.py
from robot_sim import calculate_positions  # Import the calculate_positions function
import pandas as pd  # Import pandas for exporting to Excel

# Parameters
R = 32
p_center = np.array([150, 0, 120])
phi_values = np.linspace(0, 2 * np.pi, 37)

# Calculate circle points
circle_points = np.array([p_center + R * np.array([0, np.cos(phi), np.sin(phi)]) for phi in phi_values])

# List to store the joint angles for each point
joint_angles_list = []

# Calculate joint angles for each point on the circle
for point in circle_points:
    x0, y0, z0 = point
    gamma_deg = 0
    gamma = np.deg2rad(gamma_deg)
    results_pos, results_neg = calculate_joint_angles(x0, y0, z0, gamma)
    joint_angles_list.append((results_pos, results_neg))

# Prepare data for export
data = []
for i, (results_pos, results_neg) in enumerate(joint_angles_list):
    data.append([i, 'positive', *results_pos])
    data.append([i, 'negative', *results_neg])

# Create a DataFrame
df = pd.DataFrame(data, columns=['Point', 'Solution', 'q1', 'q2', 'q3', 'q4'])

# Export to Excel
df.to_excel('./joint_angles.xlsx', index=False)

# Print the joint angles for each point
# for i, (results_pos, results_neg) in enumerate(joint_angles_list):
#     print(f"Point {i}:")
#     print("Results in radians (positive q3):")
#     print("q1: ", results_pos[0])
#     print("q2: ", results_pos[1])
#     print("q3: ", results_pos[2])
#     print("q4: ", results_pos[3])
#     print("Results in radians (negative q3):")
#     print("q1: ", results_neg[0])
#     print("q2: ", results_neg[1])
#     print("q3: ", results_neg[2])
#     print("q4: ", results_neg[3])
#     print()

# # Set up the figure for animation
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlim([-200, 200])
# ax.set_ylim([-200, 200])
# ax.set_zlim([0, 300])

# # Initialize the line object for the robot arm
# line, = ax.plot([], [], [], marker='o', color='blue', linestyle='-', linewidth=2)

# # Function to initialize the animation
# def init():
#     line.set_data([], [])
#     line.set_3d_properties([])
#     return line,

# # Function to update the animation frame by frame
# def update(frame):
#     # Use the positive q3 solution for this example
#     joint_angles = joint_angles_list[frame][0]
#     x, y, z = calculate_positions(joint_angles)
#     line.set_data(x, y)
#     line.set_3d_properties(z)
#     return line,

# # Create the animation
# ani = animation.FuncAnimation(fig, update, frames=len(joint_angles_list), init_func=init, blit=True)

# # Show the plot
# plt.show()
