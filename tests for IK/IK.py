# Python script to calculate q1, q2, q3, and q4

import numpy as np



def calculate_joint_angles(x0, y0, z0):
    # Define the lengths of the links
    a1 = 50
    a2 = 93
    a3 = 93
    a4 = 50

    q1 = np.arctan2(y0, x0)
    
    # q2 two solutions
    cos_q2 = (x0**2 + y0**2 + z0**2 - a1**2 - (a2 + a3 + a4)**2) / (2 * a1 * (a2 + a3 + a4))
    if -1 <= cos_q2 <= 1:
        q2_1 = np.arccos(cos_q2)
        q2_2 = 2 * np.pi - q2_1
    else:
        q2_1, q2_2 = np.nan, np.nan
    
    # q3 two solutions
    cos_q3 = (x0**2 + y0**2 + z0**2 - (a1 + a2)**2 - (a3 + a4)**2) / (2 * (a1 + a2) * (a3 + a4))
    if -1 <= cos_q3 <= 1:
        q3_1 = np.arccos(cos_q3)
        q3_2 = 2 * np.pi - q3_1
    else:
        q3_1, q3_2 = np.nan, np.nan
    
    # q4 two solutions
    cos_q4 = (x0**2 + y0**2 + z0**2 - (a1 + a2 + a3)**2 - a4**2) / (2 * (a1 + a2 + a3) * a4)
    if -1 <= cos_q4 <= 1:
        q4_1 = np.arccos(cos_q4)
        q4_2 = 2 * np.pi - q4_1
    else:
        q4_1, q4_2 = np.nan, np.nan
    
    return (q1, (q2_1, q2_2), (q3_1, q3_2), (q4_1, q4_2))
# Test the function with sample inputs (user can modify the values)
x0_test = 30
y0_test = 50
z0_test = 50+2+93


q1, q2, q3, q4 = calculate_joint_angles(x0_test, y0_test, z0_test)
#in radians
print("Results in radians:")
print("q1: ", q1)
print("q2: ", q2)
print("q3: ", q3)
print("q4: ", q4)

#in degrees
print("\nResults in degrees:")
print("q1: ", np.degrees(q1))
print("q2: ", np.degrees(q2))
print("q3: ", np.degrees(q3))
print("q4: ", np.degrees(q4))
