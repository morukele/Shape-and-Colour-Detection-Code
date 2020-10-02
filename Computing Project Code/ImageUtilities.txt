from collections import deque
import numpy as np
import cv2
import math


# Stacking image fuction that was open sourced
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(
                        imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(
                        imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(
                        imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(
                    imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(
                    imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def getArrowContours(img, outputImg, minArea, colour):
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > minArea:
            # Peri is the perimeter of the shape
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            if len(approx) == 9:
                M = cv2.moments(cnt)  # Calculating the moment of the arrow
                # Calculating the center of the arrow
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                center = (cX, cY)

                # Drawing the center point
                cv2.circle(outputImg, center, 5, (0, 255, 0), -1)

                # Drawing the contour of the arrow and the bounding box
                cv2.drawContours(outputImg, cnt, -1, (0, 255, 0), 7)
                x, y, w, h = cv2.boundingRect(approx)

                # Creating a min rect bounding box
                (_, _), (_, _), angle = cv2.minAreaRect(approx)
                rect = cv2.minAreaRect(approx)
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                pt1, pt2 = box[1]

                #cv2.drawContours(outputImg, [box], 0, (0, 0, 255), 5)

                cv2.rectangle(outputImg, (x, y),
                              (x + w, y + h), (0, 255, 0), 5)

                # Putting text information on the object
                cv2.putText(outputImg, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,
                            (0, 255, 0), 2)
                cv2.putText(outputImg, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                            (0, 255, 0), 2)
                cv2.putText(outputImg, "Shape: Arrow ", (x + w + 20, y + 70), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                            (0, 255, 0), 2)
                cv2.putText(outputImg, "Colour: " + colour, (x + w + 20, y + 95), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                            (0, 255, 0), 2)
                cv2.putText(outputImg, "Cx: " + str(cX) + " Cy: " + str(cY), (x + w + 20, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                            (0, 255, 0), 2)
                cv2.putText(outputImg, "Angle: " + str(int(angle)), (x + w + 20, y - 30), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                            (0, 255, 0), 2)


def getBoxContours(img, outputImg, minArea, colour):
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > minArea:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            if len(approx) == 4:
                cv2.drawContours(outputImg, cnt, -1, (0, 255, 0), 7)
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(outputImg, (x, y),
                              (x + w, y + h), (0, 255, 0), 5)
                cv2.putText(outputImg, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,
                            (0, 255, 0), 2)
                cv2.putText(outputImg, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                            (0, 255, 0), 2)
                cv2.putText(outputImg, "Shape: Box ", (x + w + 20, y + 70), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                            (0, 255, 0), 2)
                cv2.putText(outputImg, "Colour: " + colour, (x + w + 20, y + 95), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                            (0, 255, 0), 2)


def getTriangleContours(img, outputImg, minArea, colour):
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > minArea:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            if len(approx) == 3:
                cv2.drawContours(outputImg, cnt, -1, (0, 255, 0), 7)
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(outputImg, (x, y),
                              (x + w, y + h), (0, 255, 0), 5)
                cv2.putText(outputImg, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,
                            (0, 255, 0), 2)
                cv2.putText(outputImg, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                            (0, 255, 0), 2)
                cv2.putText(outputImg, "Shape: Triangle ", (x + w + 20, y + 70), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                            (0, 255, 0), 2)
                cv2.putText(outputImg, "Colour: " + colour, (x + w + 20, y + 95), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                            (0, 255, 0), 2)


def getCrossContours(img, outputImg, minArea, colour):
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > minArea:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            if len(approx) == 12:
                cv2.drawContours(outputImg, cnt, -1, (0, 255, 0), 7)
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(outputImg, (x, y),
                              (x + w, y + h), (0, 255, 0), 5)
                cv2.putText(outputImg, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,
                            (0, 255, 0), 2)
                cv2.putText(outputImg, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                            (0, 255, 0), 2)
                cv2.putText(outputImg, "Shape: Cross ", (x + w + 20, y + 70), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                            (0, 255, 0), 2)
                cv2.putText(outputImg, "Colour: " + colour, (x + w + 20, y + 95), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                            (0, 255, 0), 2)
