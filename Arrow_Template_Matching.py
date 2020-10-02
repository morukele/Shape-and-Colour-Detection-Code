import cv2
import numpy as np
import imutils
import os


# A Function that loads the tempalates from the direcotry
# The function returns a list of black and white images
def load_images_from_dir(dir):
    images = []
    for filename in os.listdir(dir):
        label = filename
        img = cv2.imread(os.path.join(dir, filename), 0)
        if img is not None:
            images.append(img)
    return images


# Loading the template and storing dimensions
tmpt_dir = './templates/'
templete = load_images_from_dir(tmpt_dir)
templete_shape = []
for tmp in templete:
    templete_shape.append(tmp.shape[:: -1])
    cv2.imshow('template', tmp)
    cv2.waitKey()
    cv2.destroyAllWindows()

#! Loading the labels into the label list
instructions = []
instructions.append('backward with free altitude')
instructions.append('foward with free altitude')
instructions.append('left with free altitude')
instructions.append('right with free altitude')

# Opening the camera
camera = cv2.VideoCapture(0)

# Defining varables
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    grabbed, feed = camera.read()
    blurr_feed = cv2.GaussianBlur(feed, (3, 3), 0)
    gray_feed = cv2.cvtColor(blurr_feed, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray_feed, 50, 200)

    # Looping throught all available templates
    for i in range(len(templete)):
        w, h = templete_shape[i]
        res = cv2.matchTemplate(gray_feed, templete[i], cv2.TM_CCOEFF_NORMED)

        threshold = 0.8
        loc = np.where(res >= threshold)
        direction = instructions[i]

        for pt in zip(*loc[::-1]):
            cv2.rectangle(feed, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 1)
            cv2.putText(
                feed, direction, (pt[0], pt[1]), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
            print(direction)

        result = cv2.hconcat([edge, gray_feed])
        cv2.imshow('Prcoess', result)
        cv2.imshow('result', feed)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
