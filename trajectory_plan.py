import numpy as np
from inverse_kinematics import calculate_joint_angles  # Assuming the IK script is named IK_2.py
from camera_transform import get_base_coordinate, compute_stylus_trajectory
import pandas as pd  # Import pandas for exporting to Excel

vector=[0.0,0.0,0.0]
stylus_vector,object_vector = get_base_coordinate(translation_vector=vector)
object_vector = [168.812, 2.919, -36.124]
trajectory = compute_stylus_trajectory(stylus_vector,object_vector,num_steps=20)
    
# List to store the joint angles for each point
joint_angles_list = []

# Calculate joint angles for each point on the circle
for point in trajectory:
    x0, y0, z0 = point
    gamma_deg = -80
    gamma = np.deg2rad(gamma_deg)
    results_pos, results_neg = calculate_joint_angles(x0, y0, z0, gamma)
    joint_angles_list.append((results_pos, results_neg))

# Prepare data for export
data = []
for i, (results_pos, results_neg) in enumerate(joint_angles_list):
    data.append([i, *results_neg])

# Create a DataFrame
df = pd.DataFrame(data, columns=['Point', 'q1', 'q2', 'q3', 'q4'])

# Export to Excel
df.to_excel('./joint_angles.xlsx', index=False)
