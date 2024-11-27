import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

img = cv2.imread('mnms.jpg')
# Changing the order from bgr to rgb so that matplotlib can show it
b,g,r = cv2.split(img)
img = cv2.merge([r,g,b])
plt.imshow(img)

# Convert the image from RGB to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

lower_thres = np.array([80,0,0])
upper_thres = np.array([255,120,100])

mask = cv2.inRange(img, lower_thres, upper_thres)

# Apply the mask to the original image
masked_frame = cv2.bitwise_and(hsv, hsv, mask=mask)

# Convert to grayscale
gray = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2GRAY)

# Apply Gaussian Blur
blurred = cv2.GaussianBlur(gray, (9, 9), 2)

min_radius = 20
max_radius = 50

circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1.5,                  # Increase dp for a better balance between accuracy and speed
    minDist=max_radius,      # Adjust based on m&m size and robots image resolution
    param1=100,              # Canny high threshold
    param2=20,               # Accumulator threshold, lower to detect more circles
    minRadius=min_radius,    # Set a smaller min radius
    maxRadius=max_radius     # Adjust max radius based on m&m size
)
img2 = img.copy()

centers = []

if circles is not None:
    print(f"Type of circles: {type(circles)}")
    # Check the shape of circles
    print(f"Shape of circles: {circles.shape}")  # Expecting shape (1, n, 3)
    
    if circles.shape[0] > 0:  # Ensure there is at least one circle detected
        circles = np.round(circles[0, :]).astype(np.int32)  # Accessing the first element
        
        for circle in circles:
            center = (circle[0], circle[1])
            centers.append(center)
            r = circle[2]

            # Draw the circle on the image
            cv2.circle(img2, center, r, (255, 255, 0), 2)
    else:
        print("No circles found in the detected array.")
    
plt.imshow(img2)
plt.show()
# "centers" contains the center of each M&M. Now, convert this to real world coordinates 