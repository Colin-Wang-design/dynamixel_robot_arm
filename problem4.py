
import sympy as sp
import openpyxl
import numpy as np
import math
import matplotlib.pyplot as plt



# Define symbols for joint angles and link lengths
theta1, theta2, theta3, theta4 = sp.symbols('theta1 theta2 theta3 theta4')
L1, L2, L3, L4 = 50, 93, 93, 50  # Lengths in mm

# Define each transformation matrix
T10 = sp.Matrix([
    [sp.cos(theta1), 0, sp.sin(theta1), 0],
    [sp.sin(theta1), 0, -sp.cos(theta1), 0],
    [0, 1, 0, L1],
    [0, 0, 0, 1]
])

T21 = sp.Matrix([
    [sp.cos(theta2 ), -sp.sin(theta2), 0, L2 * sp.cos(theta2)],
    [sp.sin(theta2), sp.cos(theta2), 0, L2 * sp.sin(theta2)],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

T32 = sp.Matrix([
    [sp.cos(theta3), -sp.sin(theta3), 0, L3 * sp.cos(theta3)],
    [sp.sin(theta3), sp.cos(theta3), 0, L3 * sp.sin(theta3)],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

T43 = sp.Matrix([
    [sp.cos(theta4), -sp.sin(theta4), 0, L4 * sp.cos(theta4)],
    [sp.sin(theta4), sp.cos(theta4), 0, L4 * sp.sin(theta4)],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

# Compute the intermediate transformations
T20 = T10 * T21
T30 = T20 * T32
T40 = T30 * T43

T20, T30, T40


# Calculate the Jacobian 

# Extract z and o vectors from transformation matrices
z0 = sp.Matrix([0, 0, 1])
z1 = T10[:3, 2]
z2 = T20[:3, 2]
z3 = T30[:3, 2]

o0 = sp.Matrix([0, 0, 0])
o1 = T10[:3, 3]
o2 = T20[:3, 3]
o3 = T30[:3, 3]
on = T40[:3, 3]

# Calculate the Jacobian matrix
Jv1 = z0.cross(on - o0)
Jv2 = z1.cross(on - o1)
Jv3 = z2.cross(on - o2)
Jv4 = z3.cross(on - o3)

Jw1 = z0
Jw2 = z1
Jw3 = z2
Jw4 = z3

J = sp.Matrix.hstack(
    sp.Matrix.vstack(Jv1, Jw1),
    sp.Matrix.vstack(Jv2, Jw2),
    sp.Matrix.vstack(Jv3, Jw3),
    sp.Matrix.vstack(Jv4, Jw4)
)

# print('Jacobian matrix:')
# print(J)


# TORQUE CALCULATION EX 9

# Load the sequence of moves from an Excel file
wb = openpyxl.load_workbook('data/test.xlsx') # _neg is elbow up configurations
sheet = wb.active

sequence_of_moves = []

for i, row in enumerate(sheet.iter_rows(min_row=2, min_col=3, max_col=6, values_only=True)):
    cleaned_row = []
    for value in row:
        if isinstance(value, str):
            # Replace non-standard minus and convert comma to dot
            value = value.replace('−', '-').replace(',', '.')
        # Convert to float
        cleaned_row.append(float(value))
    sequence_of_moves.append(cleaned_row)

print("Shape of the sequence of moves:", len(sequence_of_moves), len(sequence_of_moves[0]))


# calculate torque for each configuration in the sequence of moves

torque_vector = []

for index, goal_positions in enumerate(sequence_of_moves):
    # Substitute the goal positions into the Jacobian matrix
    # print(f"Processing point index: {index}")  # Print the index of the point being processed
    #print(" goal positions:", goal_positions)  # Print the original goal positions
    values = {theta1: goal_positions[0], theta2: goal_positions[1], theta3: goal_positions[2], theta4: goal_positions[3]}
    J_val = J.subs(values)

    # Use only the first 3 rows of the final Jacobian
    J_val_reduced = J_val[:3, :]

    # Define the force vector
    Fz = -1 # Newton
    F = sp.Matrix([0, 0, Fz,0,0,0])

    # Calculate the torque vector
    torque = J_val.T * F
    print("Torque:", torque)
    torque_vector.append(torque)



# Plot the evolution of the 4 joint angles over the variation of phi from 0 to 2pi
phi_values = np.linspace(0, 2 * np.pi, 37)
torque1_values = [torque[0] / 1000 for torque in torque_vector]
torque2_values = [torque[1] / 1000 for torque in torque_vector]
torque3_values = [torque[2] / 1000 for torque in torque_vector]
torque4_values = [torque[3] / 1000 for torque in torque_vector]

plt.figure(figsize=(10, 6))
plt.plot(phi_values, torque1_values, label='Torque1')
plt.plot(phi_values, torque2_values, label='Torque2')
plt.plot(phi_values, torque3_values, label='Torque3')
plt.plot(phi_values, torque4_values, label='Torque4')
plt.xlabel('Phi (radians)')
plt.ylabel('Torque (Nm)')
plt.title('Evolution of Joint Torques over Phi')
plt.legend()
plt.grid(True)
plt.show()


    
    
