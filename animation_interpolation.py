import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import openpyxl

# Toggle variable to activate and deactivate the animation
animate_toggle = True

# Load the sequence of moves from the Excel file
wb1 = openpyxl.load_workbook('data/test.xlsx')
sheet1 = wb1.active

sequence_of_moves1 = []

# Read every second row starting from the second row, and columns 3 to 6 for the first file
for i, row in enumerate(sheet1.iter_rows(min_row=2, min_col=3, max_col=6, values_only=True)):
    cleaned_row = []
    for value in row:
        if isinstance(value, str):
            value = value.replace('−', '-').replace(',', '.')
        cleaned_row.append(float(value))
    sequence_of_moves1.append(cleaned_row)

# Define the coefficients for the quintic interpolation functions for each joint of the robot
# Each array represents a section (A, B, C, D)
# Each row in the array represents the coefficients for a joint (q1, q2, q3, q4)
# Each column represents the order of the coefficient (from 0 to 5)

# COEFFICCIENTS WITH INITIAL AND FINAL VELOCITY = 0,0,0
coefficients_A = np.array([
    [0.210183, 0, 0, -0.082728, 0.039546, -0.005659],  # Coefficients for q1 in section A
    [1.430172, 0, 0, 0.075320, -0.056490, 0.011298],  # Coefficients for q2 in section A
    [-1.669903, 0, 0, 0.349555, -0.262166, 0.052433],  # Coefficients for q3 in section A
    [0.239730, 0, 0, -0.424875, 0.318656, -0.063731]   # Coefficients for q4 in section A
])

coefficients_B = np.array([
    [0, -0.180000, 0, 0.007272, 0.017046, -0.005659],  # Coefficients for q1 in section B
    [1.490428, 0, 0, -0.033422, 0.019829, -0.003442],  # Coefficients for q2 in section B
    [-1.390259, 0, 0, -0.216739, 0.145952, -0.027530], # Coefficients for q3 in section B
    [-0.100169, 0, 0, 0.463564, -0.352509, 0.070985]   # Coefficients for q4 in section B
])

coefficients_C = np.array([
    [-0.210183, 0, 0, 0.082728, -0.039546, 0.005659],  # Coefficients for q1 in section C
    [1.430172, -0.041898, 0, -0.073414, 0.060298, -0.012583],  # Coefficients for q2 in section C
    [-1.669903, -0.132816, 0, -0.108441, 0.097932, -0.021247], # Coefficients for q3 in section C
    [0.239730, -0.038690, 0, 0.501959, -0.371633, 0.073843]    # Coefficients for q4 in section C
])

coefficients_D = np.array([
    [0, 0.180000, 0, -0.007272, -0.017046, 0.005659],  # Coefficients for q1 in section D
    [1.321164, 0, 0, 0.136260, -0.102195, 0.020439],   # Coefficients for q2 in section D
    [-1.916034, 0, 0, 0.307665, -0.230748, 0.046150],  # Coefficients for q3 in section D
    [0.594870, 0, 0, -0.443925, 0.332944, -0.066589]   # Coefficients for q4 in section D
])

# COEFICCIENTS WITH INITIAL AND FINAL VELOCITY = 0,0,27
# coefficients_A = np.array([
#     [0.210183, 0, 0, -0.082728, 0.039546, -0.005659],  # Coefficients for q1 in section A
#     [1.430172, 0.041898, 0, 0.012473, -0.014592, 0.003442],  # Coefficients for q2 in section A
#     [-1.669903, 0.132816, 0, 0.150331, -0.129350, 0.027530],  # Coefficients for q3 in section A
#     [0.239730, 0.038690, 0, -0.482909, 0.357345, -0.070985]   # Coefficients for q4 in section A
# ])

# coefficients_B = np.array([
#     [0, -0.180000, 0, 0.007272, 0.017046, -0.005659],  # Coefficients for q1 in section B
#     [1.490428, 0, 0, -0.033422, 0.019829, -0.003442],  # Coefficients for q2 in section B
#     [-1.390259, 0, 0, -0.216739, 0.145952, -0.027530], # Coefficients for q3 in section B
#     [-0.100169, 0, 0, 0.463564, -0.352509, 0.070985]   # Coefficients for q4 in section B
# ])

# coefficients_C = np.array([
#     [-0.210183, 0, 0, 0.082728, -0.039546, 0.005659],  # Coefficients for q1 in section C
#     [1.430172, -0.041898, 0, -0.073414, 0.060298, -0.012583],  # Coefficients for q2 in section C
#     [-1.669903, -0.132816, 0, -0.108441, 0.097932, -0.021247], # Coefficients for q3 in section C
#     [0.239730, -0.038690, 0, 0.501959, -0.371633, 0.073843]    # Coefficients for q4 in section C
# ])

# coefficients_D = np.array([
#     [0, 0.180000, 0, -0.007272, -0.017046, 0.005659],  # Coefficients for q1 in section D
#     [1.321164, 0, 0, 0.094363, -0.065535, 0.012583],   # Coefficients for q2 in section D
#     [-1.916034, 0, 0, 0.174849, -0.114534, 0.021247],  # Coefficients for q3 in section D
#     [0.594870, 0, 0, -0.482615, 0.366797, -0.073843]   # Coefficients for q4 in section D
# ])



# Define the number of timesteps for each section
t = 9

# Create a linear space to evaluate the function in t timesteps
timesteps = np.linspace(0, 2, t) #goes from 0 to 2 in t steps

# Function to evaluate the quintic polynomial
def evaluate_quintic(coefficients, t):
    return np.polyval(coefficients[::-1], t)

# Create an array to store the joint configurations for each timestep
joint_configurations = []

# Evaluate the quintic polynomial for each section and each joint
for section in [coefficients_A, coefficients_B, coefficients_C, coefficients_D]:
    for t in timesteps:
        joint_configuration = []
        for joint_coefficients in section:
            joint_configuration.append(evaluate_quintic(joint_coefficients, t))
        joint_configurations.append(joint_configuration)

joint_configurations = np.array(joint_configurations)

# Save the coordinates of all the interpolated points
interpolated_points = []

# Function to compute the forward kinematics
def forward_kinematics(q1, q2, q3, q4):
    L1, L2, L3, L4 = 0.50, 0.93, 0.93, 0.50  # Link lengths
    x0, y0, z0 = 0, 0, 0
    x1, y1, z1 = 0, 0, L1
    x2, y2, z2 = L2 * np.cos(q1) * np.cos(q2), L2 * np.sin(q1) * np.cos(q2), L1 + L2 * np.sin(q2)
    x3, y3, z3 = x2 + L3 * np.cos(q1) * np.cos(q2 + q3), y2 + L3 * np.sin(q1) * np.cos(q2 + q3), z2 + L3 * np.sin(q2 + q3)
    x4, y4, z4 = x3 + L4 * np.cos(q1) * np.cos(q2 + q3 + q4), y3 + L4 * np.sin(q1) * np.cos(q2 + q3 + q4), z3 + L4 * np.sin(q2 + q3 + q4)
    return np.array([[x0, y0, z0], [x1, y1, z1], [x2, y2, z2], [x3, y3, z3], [x4, y4, z4]])

# Create the animation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set equal scaling
ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

# Set limits
ax.set_xlim([-0.5, 2.5])
ax.set_ylim([-1, 1])
ax.set_zlim([0, 2])

# Add axis labels
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

line1, = ax.plot([], [], [], 'o-', lw=2, color='blue', label='Test Data')
line2, = ax.plot([], [], [], 'o-', lw=2, color='red', label='Interpolated Data')
scatter1 = ax.scatter([], [], [], color='blue', s=10, alpha=1.0)
scatter2 = ax.scatter([], [], [], color='red', s=10, alpha=1.0)

end_effector_positions1 = []
end_effector_positions2 = []

def init():
    line1.set_data([], [])
    line1.set_3d_properties([])
    line2.set_data([], [])
    line2.set_3d_properties([])
    scatter1._offsets3d = ([], [], [])
    scatter2._offsets3d = ([], [], [])
    return line1, line2, scatter1, scatter2

def animate(i):
    if i < len(sequence_of_moves1):
        angles1 = sequence_of_moves1[i]
        positions1 = forward_kinematics(*angles1)
        line1.set_data(positions1[:, 0], positions1[:, 1])
        line1.set_3d_properties(positions1[:, 2])
        end_effector_positions1.append(positions1[-1])
        scatter1._offsets3d = (np.array(end_effector_positions1)[:, 0], np.array(end_effector_positions1)[:, 1], np.array(end_effector_positions1)[:, 2])
    
    if i < len(joint_configurations):
        angles2 = joint_configurations[i]
        positions2 = forward_kinematics(*angles2)
        line2.set_data(positions2[:, 0], positions2[:, 1])
        line2.set_3d_properties(positions2[:, 2])
        end_effector_positions2.append(positions2[-1])
        scatter2._offsets3d = (np.array(end_effector_positions2)[:, 0], np.array(end_effector_positions2)[:, 1], np.array(end_effector_positions2)[:, 2])
        interpolated_points.append(positions2[-1])
    
    return line1, line2, scatter1, scatter2

if animate_toggle:
    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=max(len(sequence_of_moves1), len(joint_configurations)), interval=200, blit=True)
    plt.legend()
    plt.show()

# Save the coordinates of all the interpolated points into a variable
interpolated_points = np.array(interpolated_points) * 100

# Plot a 3D circle in the yz space with radius 32mm and center at (150, 0, 120)
fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
circle_radius = 32
circle_center = (150, 0, 120)  # Center in 3D space

# Create points for the circle in the yz-plane
theta = np.linspace(0, 2 * np.pi, 100)
circle_y = circle_center[1] + circle_radius * np.cos(theta)
circle_z = circle_center[2] + circle_radius * np.sin(theta)
circle_x = np.full_like(circle_y, circle_center[0])

# Plot the circle
ax3.plot(circle_x, circle_y, circle_z, 'g', lw=1)

# Plot the positions of the 4 knot points in the top, left, bottom, and right parts of the circle
knot_points = np.array([
    [circle_center[0], circle_center[1] + circle_radius, circle_center[2]],  # Top
    [circle_center[0], circle_center[1] - circle_radius, circle_center[2]],  # Bottom
    [circle_center[0], circle_center[1], circle_center[2] - circle_radius],  # Left
    [circle_center[0], circle_center[1], circle_center[2] + circle_radius]   # Right
])
ax3.scatter(knot_points[:, 0], knot_points[:, 1], knot_points[:, 2], color='red')

# Draw a line connecting all the interpolated timestep points
if len(interpolated_points) > 0:
    print("Interpolated points shape:", interpolated_points.shape)
    print("Interpolated points:", interpolated_points)
    ax3.plot(interpolated_points[:, 0], interpolated_points[:, 1], interpolated_points[:, 2], '-b')  # Plot interpolated points in 3D space

# Set axis limits and labels
ax3.set_xlim([140, 160])
ax3.set_ylim([-35, 35])
ax3.set_zlim([85, 155])
ax3.set_xlabel('X axis')
ax3.set_ylabel('Y axis')
ax3.set_zlabel('Z axis')

# Set equal scaling
ax3.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

# Add legend
ax3.legend(['Circle', 'Knot Points', 'Interpolated Points'])

plt.show()