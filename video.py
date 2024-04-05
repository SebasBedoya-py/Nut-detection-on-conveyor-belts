import cv2
import numpy as np

cap = cv2.VideoCapture(1)

while True:
    _, img = cap.read()

    
    # Image Processing -------------------------------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_filter = np.array([40,20,100])
    upper_filter = np.array([100,255,255])
    mask = cv2.inRange(hsv, lower_filter, upper_filter)
    bit = cv2.bitwise_and(img, img, mask = mask)
    median = cv2.medianBlur(bit, 5)
    gray= cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)

    # Find Contours -----------------------------------
    _, thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
    contours, huerarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(img, contours, -1, (0,0,255), 2)

    # Find Area ---------------------------------------
    for cnt in contours:
        epsilon = 0.047 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        area = cv2.contourArea(cnt)

        if len(approx) == 6: 
            (x, y, w, h) = cv2.boundingRect(cnt)

            if area > 3100 and area <3800:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, 'OK', (x, y), 1, 2, (0, 255, 0))
            elif area > 5800 and area <6200:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, 'OK', (x, y), 1, 2, (0, 255, 0))

            else:
                if area > 1000:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(img, 'NOK', (x, y), 1, 2, (0, 0, 255))
        if len(approx) > 6: 
            if area > 1000:
                (x, y, w, h) = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(img, 'NOK', (x, y), 1, 2, (0, 0, 255))
            
        

    
    # cv2.imshow("Deteccion2", gray)
    cv2.imshow("Deteccion", img)

    
    key = cv2.waitKey(1)
    if key == 27:
        break


cap.release()
cv2.destroyAllWindows()