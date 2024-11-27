import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IK_2 import calculate_joint_angles  # Assuming the IK script is named IK_2.py
from robot_sim import calculate_positions  # Import the calculate_positions function
import pandas as pd  # Import pandas for exporting to Excel

# Parameters
R = 32
p_center = np.array([150, 0, 120])
phi_values = np.linspace(0, 2 * np.pi, 37)

# Plot the circle points in a 3D window
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
circle_points = np.array([p_center + R * np.array([0, np.cos(phi), np.sin(phi)]) for phi in phi_values])

for i, point in enumerate(circle_points):
    ax.scatter(point[0], point[1], point[2], color='red')
    ax.text(point[0], point[1], point[2], f'{i+1}', color='black')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Circle Points')
plt.show()

# Calculate circle points
circle_points = np.array([p_center + R * np.array([0, np.cos(phi), np.sin(phi)]) for phi in phi_values])

# List to store the joint angles for each point
joint_angles_list = []

# Calculate joint angles for each point on the circle
for point in circle_points:
    x0, y0, z0 = point
    gamma_deg = 0
    gamma = np.deg2rad(gamma_deg)
    print("gamma: ", gamma)
    print("x0: ", x0)
    print("y0: ", y0)
    print("z0: ", z0)

    results_pos, results_neg = calculate_joint_angles(x0, y0, z0, gamma)
    print("results_pos: ", results_pos)
    print("results_neg: ", results_neg)
    joint_angles_list.append((results_pos, results_neg))

# Prepare data for export
data = []
for i, (results_pos, results_neg) in enumerate(joint_angles_list):
    data.append([i+1, 'positive', *results_pos])
    data.append([i+1, 'negative', *results_neg])

# Create a DataFrame
df = pd.DataFrame(data, columns=['Point', 'Solution', 'q1', 'q2', 'q3', 'q4'])

# Export to Excel
output_dir = 'tests for IK'
output_path = os.path.join(output_dir, 'joint_angles.xlsx')
df.to_excel(output_path, index=False)

# Print the joint angles for each point
for i, (results_pos, results_neg) in enumerate(joint_angles_list):
    print(f"Point {i+1}:")
    print("Results in radians (positive q3):")
    print("q1: ", results_pos[0])
    print("q2: ", results_pos[1])
    print("q3: ", results_pos[2])
    print("q4: ", results_pos[3])
    print("Results in radians (negative q3):")
    print("q1: ", results_neg[0])
    print("q2: ", results_neg[1])
    print("q3: ", results_neg[2])
    print("q4: ", results_neg[3])
    print()

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

a1 = 50
a2 = 93
a3 = 93
a4 = 50

# Calculate the end effector positions using the positive joint angles
end_effector_positions = []

for results_neg, _ in joint_angles_list:
    q1, q2, q3, q4 = results_pos
    x = a2 * np.cos(q2) + a3 * np.cos(q2 + q3) + a4 * np.cos(q2 + q3 + q4)
    z = a1 + a2 * np.sin(q2) + a3 * np.sin(q2 + q3) + a4 * np.sin(q2 + q3 + q4)
    y = x * np.tan(q1)
    end_effector_positions.append([x, y, z])

# Convert to numpy array for easier manipulation
end_effector_positions = np.array(end_effector_positions)

# Plot the end effector positions in a 3D window
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i, point in enumerate(end_effector_positions):
    ax.scatter(point[0], point[1], point[2], color='blue')
    ax.text(point[0], point[1], point[2], f'{i+1}', color='black')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('End Effector Circle Points')
plt.show()



