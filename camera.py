import cv2
import numpy as np

####### Resources #################
'''
https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
https://www.w3schools.com/colors/colors_rgb.asp
'''
###################################
cap = cv2.VideoCapture(0)

lower_color = np.array([90, 200,200]) 
upper_color = np.array([160, 255, 255])  


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while True:
    
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []

    for contour in contours:
        if cv2.contourArea(contour) > 100: 
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, x + w, y + h))


    for box in bounding_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    out.write(frame)

    cv2.imshow('Object Tracking', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to exit
        break

cap.release()
out.release()

cv2.destroyAllWindows()





