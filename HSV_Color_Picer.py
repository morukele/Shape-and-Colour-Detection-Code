import cv2
import numpy as np


def empty(x):
    pass


cv2.namedWindow("Parameters")
cv2.createTrackbar("L_H", "Parameters", 0, 180, empty)
cv2.createTrackbar("L_S", "Parameters", 0, 255, empty)
cv2.createTrackbar("L_V", "Parameters", 0, 255, empty)
cv2.createTrackbar("U_H", "Parameters", 180, 180, empty)
cv2.createTrackbar("U_S", "Parameters", 255, 255, empty)
cv2.createTrackbar("U_V", "Parameters", 255, 255, empty)

cap = cv2.VideoCapture(0)

while True:
    grabbed, feed = cap.read()

    hsv = cv2.cvtColor(feed, cv2.COLOR_BGR2HSV)

    L_H = cv2.getTrackbarPos("L_H", "Parameters")
    L_S = cv2.getTrackbarPos("L_S", "Parameters")
    L_V = cv2.getTrackbarPos("L_V", "Parameters")
    U_H = cv2.getTrackbarPos("U_H", "Parameters")
    U_S = cv2.getTrackbarPos("U_S", "Parameters")
    U_V = cv2.getTrackbarPos("U_V", "Parameters")

    lower_blue = np.array([L_H, L_S, L_V], np.uint8)
    upper_blue = np.array([U_H, U_S, U_V], np.uint8)

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    cv2.imshow("Mask", mask)
    cv2.imshow("Output", feed)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
