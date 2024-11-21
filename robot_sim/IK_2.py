# Python script to calculate q1, q2, q3, and q4

import numpy as np

def wrap_angle(angle):
    # If the angle is greater than pi/2, subtract pi to bring it within [-pi, pi]
    if angle > np.pi/2:
        angle -= np.pi

    return angle


def calculate_joint_angles(x0, y0, z0, gamma):
    # Define the lengths of the links
    a1 = 50
    a2 = 93
    a3 = 93
    a4 = 50

    # Calculate q1 based on the value of y0
    epsilon = 1e-9  # Small value to avoid division by zero
    if y0 > 0:
        q1 = np.arctan2(y0, x0 + epsilon)
    elif y0 < 0:
        q1 = np.arctan2(y0, x0 + epsilon) + np.pi
    else:  # y0 == 0
        q1 = 0 if x0 >= 0 else np.pi
    
    q1 = wrap_angle(q1)

    q3_in = ((np.sqrt(x0**2+y0**2)-a4*np.cos(gamma))**2 + (z0-a1-a4*np.sin(gamma))**2 -(a2**2+a3**2)) / (2*a2*a3)
    # print("q3_in: ", q3_in)
    # # in degrees
    # print("q3_in in degrees: ", np.degrees(q3_in))
    
    q3_pos = np.arccos(q3_in)
    q3_neg = -np.arccos(q3_in)

    def calculate_q2_q4(q3):
        q2_in1 = (z0-a1-a4*np.sin(gamma)) #d_2
        q2_in2 = (np.sqrt(x0**2+y0**2)-a4*np.cos(gamma)) #r_1
        q2_in3 = a3*np.sin(q3)
        q2_in4 = a2+a3*np.cos(q3)
        # print("q2_in1: ", q2_in1)
        # print("q2_in2: ", q2_in2)
        # print("q2_in3: ", q2_in3)
        # print("q2_in4: ", q2_in4)

        q2 = np.arctan2(q2_in1, q2_in2) - np.arctan2(q2_in3, q2_in4)
        q4 = gamma - q2 - q3
        return q2, q4

    q2_pos, q4_pos = calculate_q2_q4(q3_pos)
    q2_neg, q4_neg = calculate_q2_q4(q3_neg)

    return (q1, q2_pos, q3_pos, q4_pos), (q1, q2_neg, q3_neg, q4_neg)

# Test the function with sample inputs (user can modify the values)
x0_test = 0
y0_test = -236
z0_test = 50
gamma_deg = 0
gamma = np.deg2rad(gamma_deg)




results_pos, results_neg = calculate_joint_angles(x0_test, y0_test, z0_test, gamma)

# # in radians
# print("Results in radians (positive q3):")
# print("q1: ", results_pos[0])
# print("q2: ", results_pos[1])
# print("q3: ", results_pos[2])
# print("q4: ", results_pos[3])

# print("\nResults in radians (negative q3):")
# print("q1: ", results_neg[0])
# print("q2: ", results_neg[1])
# print("q3: ", results_neg[2])
# print("q4: ", results_neg[3])

# in degrees
print("\nResults in degrees (positive q3):")
print("q1: ", np.degrees(results_pos[0]))
print("q2: ", np.degrees(results_pos[1]))
print("q3: ", np.degrees(results_pos[2]))
print("q4: ", np.degrees(results_pos[3]))

print("\nResults in degrees (negative q3):")
print("q1: ", np.degrees(results_neg[0]))
print("q2: ", np.degrees(results_neg[1]))
print("q3: ", np.degrees(results_neg[2]))
print("q4: ", np.degrees(results_neg[3]))
