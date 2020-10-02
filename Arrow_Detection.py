import cv2
import numpy as np
import imutils
import os

#! Loading the labels into the label list
instructions = []
instructions.append('backward with free altitude')
instructions.append('foward with free altitude')
instructions.append('left with free altitude')
instructions.append('right with free altitude')


# Opening the camera
camera = cv2.VideoCapture(0)

# getting the width and height of the video feed
# Returns a float, had to cast it to type integer
w_v = int(camera.get(3))
h_v = int(camera.get(4))

p = 200  # padding for the cropping of the image.

print("height, width : ", h_v, w_v)

# Defining varables
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    grabbed, feed = camera.read()
    blurr_feed = cv2.GaussianBlur(feed, (3, 3), 0)

    #! Cropped the video since we are only interested in the center and reduce noise
    # Creating a copy to show the output
    cropped = feed[p:h_v, p:w_v]
    feed_copy = np.copy(cropped)

    #! Creating color mask
    # Converting to HSV becuase it is less sensitive to illumination
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

    # Creating the range for red in HSV
    # HSV has two domains for red because it exist between 150-180-0-30 degrees in the HSV Cylinder spectrum
    # Lower Red Range
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # Upper red range
    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Generating the final red mask
    red_mask = mask1 + mask2

    #! Dilating and Erroding the resulting color mask
    kernel = np.ones((5, 5), np.uint8)
    # ? Erode
    e1 = cv2.erode(red_mask, kernel, iterations=2)
    # ? Dilating
    d1 = cv2.dilate(e1, kernel, iterations=2)

    #! Finding the center of the arrow using the moment
    M = cv2.moments(d1)
    if M["m00"] != 0:
        cX = int(M["m10"]/M["m00"])
        cY = int(M["m01"]/M["m00"])

        # Drawing the center point on the image
        cv2.circle(feed_copy, (cX, cY), 5, (0, 0, 255), -1)

    #! Detecting the edges using canny edges
    lower_thres = 50
    upper_thres = 200
    edges = cv2.Canny(d1, lower_thres, upper_thres)

    #! Creating a bounding rectangle that doesn't account for rotation
    x, y, w, h = cv2.boundingRect(edges)
    if x and y and w and h is not None:
        endx, endy = x+w, y+h
        cv2.rectangle(feed_copy, (x, y), (endx, endy), (0, 255, 0), 2)

        # ? Project a point from the center of the circle and draw a line
        cv2.circle(feed_copy, (cX, y + 10), 5, (255, 0, 0), -1)
        cv2.line(feed_copy, (cX, cY), (cX, y+10), (255, 0, 0), 5)

    #! Corner Detection using the dialated image
    max_corners = 10
    quality_level = 0.01
    min_distance = 10
    corners = cv2.goodFeaturesToTrack(
        d1, max_corners, quality_level, min_distance)

    # Drawing the corners in the arrow
    if corners is not None:
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(cropped, (x, y), 5, (255, 0, 0), -1)

    #! Displaying result
    c1 = np.concatenate((cropped, feed_copy), axis=1)
    cv2.imshow("Original Feed and Arrow", c1)
    cv2.imshow("Red Mask", red_mask)
    cv2.imshow("Dilated", d1)
    cv2.imshow("Edges", edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
