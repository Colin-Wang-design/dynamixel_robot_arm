#!/usr/bin/python
#
# ENGR421 -- Applied Robotics, Spring 2013
# OpenCV Python Demo
# Taj Morton <mortont@onid.orst.edu>
#
import sys
import cv2
import time
import numpy
import os
 
##
# Opens a video capture device with a resolution of 800x600
# at 30 FPS.
##
def open_camera(cam_id = 1):
    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FPS, 24)
    return cap
 
##
# Gets a frame from an open video device, or returns None
# if the capture could not be made.
##
def get_frame(device):
    ret, img = device.read()
    if (ret == False): # failed to capture
        print("Error capturing from video device.")
        return None
    return img
 
##
# Closes all OpenCV windows and releases video capture device before exit.
##
def cleanup(cam_id = 0): 
    cv2.destroyAllWindows()
    cv2.VideoCapture(cam_id).release()
 
##
# Creates a new RGB image of the specified size, initially
# filled with black.
##
def new_rgb_image(width, height):
    image = numpy.zeros( (height, width, 3), numpy.uint8)
    return image
##
# Converts an RGB image to grayscale, where each pixel
# now represents the intensity of the original image.
##
def rgb_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
 
##
# Converts an image into a binary image at the specified threshold.
# All pixels with a value <= threshold become 0, while
# pixels > threshold become 1
def do_threshold(image, threshold = 170):
    (thresh, im_bw) = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return (thresh, im_bw)

def find_contours(image):
    (contours, hierarchy) = cv2.findContours(image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    return contours

def find_centers(contours):
    centers = []
    for contour in contours:
        moments = cv2.moments(contour, True)
 
        # If any moment is 0, discard the entire contour. This is
        # to prevent division by zero.
        if (len(filter(lambda x: x==0, moments.values())) > 0): 
            continue
 
        center = (moments['m10']/moments['m00'] , moments['m01']/moments['m00'])
        # Convert floating point contour center into an integer so that
        # we can display it later.
 
        center = map(lambda x: int(round(x)), center)
        centers.append(center)
    return centers

def draw_centers(centers, image):
    for center in centers:
        cv2.circle(image, tuple(center), 20, (255,255,0), 2)
########### Main Program ###########

if __name__ == "__main__":
    # Camera ID to read video from (numbered from 0)
    camera_id = 0
    dev = open_camera(camera_id) # open the camera as a video capture device
 
    while True:
        img_orig = get_frame(dev) # Get a frame from the camera
        if img_orig is not None: # if we did get an image
            cv2.imshow("video", img_orig) # display the image in a window named "video"
            img_gray = rgb_to_gray(img_orig) # Convert img_orig from video camera from RGB to Grayscale
 
            # Converts grayscale image to a binary image with a threshold value of 220. Any pixel with an
            # intensity of <= 220 will be black, while any pixel with an intensity > 220 will be white:
            (thresh, img_threshold) = do_threshold(img_gray, 220)
            
            # cv2.imshow("Grayscale", img_gray)
            # cv2.imshow("Threshold", img_threshold)
            # the following code will find the contours of your image:
            contours = find_contours(img_threshold)
            
            # Here, we are creating a new RBB image to display our results on
            results_image = new_rgb_image(img_threshold.shape[1], img_threshold.shape[0])
            color = (0, 0, 255)
            cv2.drawContours(results_image, contours, -1, color, 2)
            
            # Display Results
            cv2.imshow("results", results_image)
        else: # if we failed to capture (camera disconnected?), then quit
            break
 
        if (cv2.waitKey(2) >= 0): # If the user presses any key, exit the loop
            break
        
 
    cleanup(camera_id) # close video device and windows before we exit