import cv2
import numpy as np


####### Resources #################
'''
https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html
https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/
https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097
https://docs.opencv.org/3.4/index.html
Previous Code
'''
###################################
def dominant_color(image, rect):
    k = 3
    x, y, w, h = rect
    cropImage = image[y:y + h, x:x + w]
    pixels = cropImage.reshape((-1, 3))
    pixels = np.float32(pixels)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
    ret , label, center = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    dominant_color = np.uint8(center[0])

    return dominant_color


# Capture video from the default camera (change 0 to the camera index if needed)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break
        
    central_rect = (300, 160, 100, 100)
    color = dominant_color(frame, central_rect)
    x, y, w, h = central_rect
    
    color_block = np.zeros((100, 100, 3), dtype=np.uint8)
    color_block[:, :] = color
    
    frame[10:110, 10:110] = color_block
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(frame, f"Dominant Color: {color}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # Break the loop if 'q' key is pressed
    cv2.imshow("Video Feed", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()