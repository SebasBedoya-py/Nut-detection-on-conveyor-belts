import cv2
import numpy as np

img = cv2.imread('C:/Users/sebas/OneDrive/Escritorio/Mechasoft/Tercol/Source/conveyor.jpg')
img = cv2.resize(img, (884 //3, 1599 //3))

# Image Processing -------------------------------
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_filter = np.array([40,0,100])
upper_filter = np.array([100,255,255])
mask = cv2.inRange(hsv, lower_filter, upper_filter)
bit = cv2.bitwise_and(img, img, mask = mask)
median = cv2.medianBlur(bit, 5)
gray= cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)

# Find Contours -----------------------------------
_, thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
contours, huerarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0,0,255), 2)

# Find Area ---------------------------------------
for cnt in contours:
    epsilon = 0.047 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    if len(approx) == 6: 
        (x, y, w, h) = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        cv2.putText(img, str(area), (x+w, y), 7, 0.5, (255, 0, 0))




cv2.imshow('Mask', thresh)
cv2.imshow('HSV_Gray', gray)
cv2.imshow('Original', img)



cv2.waitKey(0)
cv2.destroyAllWindows()