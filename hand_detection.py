import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('face.xml')
video = cv2.VideoCapture(0)
hueLower=3
hueUpper=33
mask = np.zeros((720, 1280), np.uint8)
bg = np.zeros((1, 65), np.float64)
fg = np.zeros((1, 65), np.float64)
#fgbg = cv2.createBackgroundSubtractorMOG2()

while video.isOpened():
    ret, img2 = video.read()

    img = cv2.GaussianBlur(img2, (11, 11), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.inRange(img, (3, 50, 50), (15, 255, 255))
    img = cv2.erode(img, None, iterations=1)
    img = cv2.medianBlur(img, 11)
    faces = face_cascade.detectMultiScale(img2, 1.3, 5)

    if faces!=():
        for face in faces:
            (x, y, w, h) = face
            img[y-50:y+h, x:x+w] = 0
    '''_, contours, heirarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) > 4000:
            cv2.drawContours(img2, c, -1, 1, 3)'''
    img2 = cv2.bitwise_and(img2, img2, mask=img)



    cv2.imshow('hand', img2)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()