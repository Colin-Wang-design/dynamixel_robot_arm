import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, TextBox
import matplotlib.pyplot as plt

# Define the lengths of the links
link_lengths = [50, 93, 93, 50]

# Function to calculate the positions of the joints using DH parameters
def calculate_positions(angles):
    x = [0]
    y = [0]
    z = [0]
    
    # Base joint (revolves around z-axis)
    theta1 = np.deg2rad(angles[0])
    d1 = link_lengths[0]
    x.append(x[-1])
    y.append(y[-1])
    z.append(z[-1] + d1)
    
    # First revolute joint
    theta2 = np.deg2rad(angles[1])
    a2 = link_lengths[1]
    x.append(x[-1] + a2 * np.cos(theta1) * np.cos(theta2))
    y.append(y[-1] + a2 * np.sin(theta1) * np.cos(theta2))
    z.append(z[-1] + a2 * np.sin(theta2))
    
    # Second revolute joint
    theta3 = np.deg2rad(angles[2] + angles[1])
    a3 = link_lengths[2]
    x.append(x[-1] + a3 * np.cos(theta1) * np.cos(theta3))
    y.append(y[-1] + a3 * np.sin(theta1) * np.cos(theta3))
    z.append(z[-1] + a3 * np.sin(theta3))
    
    # Third revolute joint
    theta4 = np.deg2rad(angles[3] + angles[2] + angles[1])
    a4 = link_lengths[3]
    x.append(x[-1] + a4 * np.cos(theta1) * np.cos(theta4))
    y.append(y[-1] + a4 * np.sin(theta1) * np.cos(theta4))
    z.append(z[-1] + a4 * np.sin(theta4))
    
    return x, y, z

# Function to calculate the angle of the last link with the ground
def calculate_last_link_angle(angles):
    theta4 = np.deg2rad(angles[3] + angles[2] + angles[1])
    return np.rad2deg(theta4)

# Initial angles
angles = [0, 0, 0, 0]

# Create the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create sliders for the angles
axcolor = 'lightgoldenrodyellow'
ax_base = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_angle1 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_angle2 = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
ax_angle3 = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)

s_base = Slider(ax_base, 'Base', 0, 360, valinit=0)
s_angle1 = Slider(ax_angle1, 'Angle 1', -180, 180, valinit=0)
s_angle2 = Slider(ax_angle2, 'Angle 2', -180, 180, valinit=0)
s_angle3 = Slider(ax_angle3, 'Angle 3', -180, 180, valinit=0)

# Create a TextBox for displaying the coordinates and angle
ax_coords = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
coords_text_box = TextBox(ax_coords, 'End Effector Coordinates and Angle')

# Update function for the sliders
def update(val):
    angles[0] = s_base.val
    angles[1] = s_angle1.val
    angles[2] = s_angle2.val
    angles[3] = s_angle3.val
    x, y, z = calculate_positions(angles)
    last_link_angle = calculate_last_link_angle(angles)
    ax.cla()
    ax.plot(x, y, z, marker='o')
    ax.set_xlim([-200, 200])
    ax.set_ylim([-200, 200])
    ax.set_zlim([0, 300])
    # Update the coordinates and angle in the TextBox
    coords_text_box.set_val(f'({x[-1]:.2f}, {y[-1]:.2f}, {z[-1]:.2f}), Angle: {last_link_angle:.2f}°')
    plt.draw()

s_base.on_changed(update)
s_angle1.on_changed(update)
s_angle2.on_changed(update)
s_angle3.on_changed(update)

# Initial plot
x, y, z = calculate_positions(angles)
last_link_angle = calculate_last_link_angle(angles)
ax.plot(x, y, z, marker='o')
ax.set_xlim([-200, 200])
ax.set_ylim([-200, 200])
ax.set_zlim([0, 300])
# Set initial coordinates and angle in the TextBox
coords_text_box.set_val(f'({x[-1]:.2f}, {y[-1]:.2f}, {z[-1]:.2f}), Angle: {last_link_angle:.2f}°')

plt.show()