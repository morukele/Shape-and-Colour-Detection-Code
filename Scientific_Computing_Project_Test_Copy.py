import cv2
import numpy as np
import ImageUtilities as iu

# Opening the camera
camera = cv2.VideoCapture(0)

# get the height and width of the camera feed
w_v = int(camera.get(3))
h_v = int(camera.get(4))

print("height and width: ", h_v, w_v)

while True:
    grabbed, feed = camera.read()
    outputImg = feed.copy()
    gray_feed = cv2.cvtColor(feed, cv2.COLOR_BGR2GRAY)

    blurred_feed = cv2.GaussianBlur(feed, (7, 7), 1)  # Blur the image
    hsv = cv2.cvtColor(blurred_feed, cv2.COLOR_BGR2HSV)  # convert to HSV

    # Create the red colour mask
    # Lower Red Range
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # Upper red range
    lower_red = np.array([170, 100, 100])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Generating the final red mask
    red_mask = mask1 + mask2

    # Create the yellow colour mask
    lower_yellow = np.array([20, 100, 100], np.uint8)
    upper_yellow = np.array([30, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Create the blue colour mask
    lower_blue = np.array([72, 66, 88])
    upper_blue = np.array([120, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((3, 3), np.uint8)

    # Eroding the color mask
    #blue_edges = cv2.erode(blue_mask, kernel, iterations=1)
    yellow_edges = cv2.erode(yellow_mask, kernel, iterations=1)
    red_edges = cv2.erode(red_mask, kernel, iterations=1)

    # Detecting the edges of the images
    lower_thres = 50
    upper_thres = 150

    red_edges = cv2.Canny(red_mask, lower_thres, upper_thres)
    blue_edges = cv2.Canny(blue_mask, lower_thres, upper_thres)
    yellow_edges = cv2.Canny(yellow_mask, lower_thres, upper_thres)
    black_edges = cv2.Canny(gray_feed, lower_thres, upper_thres)

    # Dilating the resulting edges
    # Dilating
    blue_edges = cv2.dilate(blue_edges, kernel, iterations=1)
    yellow_edges = cv2.dilate(yellow_edges, kernel, iterations=1)
    red_edges = cv2.dilate(red_edges, kernel, iterations=1)

    # Detecting the contours and displaying them.
    iu.getArrowContours(blue_edges, outputImg, 2000, "blue")
    iu.getArrowContours(red_edges, outputImg, 2000, "red")
    iu.getArrowContours(yellow_edges, outputImg, 2000, "yellow")
    iu.getBoxContours(red_edges, outputImg, 2000, "red")
    iu.getTriangleContours(red_edges, outputImg, 2000, "red")
    iu.getCrossContours(black_edges, outputImg, 2000, "Black")

    # Image stacking script that was open sourced
    imgStack = iu.stackImages(
        0.5, ([feed, gray_feed, hsv], [
            red_mask, yellow_mask, blue_mask],
            [red_edges, yellow_edges, blue_edges],
            [black_edges, outputImg, outputImg]))

    # Displays the results
    cv2.imshow("Result", imgStack)
    cv2.imshow("Output", outputImg)

    # Closes the camera
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
