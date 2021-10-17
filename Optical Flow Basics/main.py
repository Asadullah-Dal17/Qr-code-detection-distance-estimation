# importing modules 

import cv2 as cv
import numpy as np
import AiPhile
import time
# point seletctor function, which let's select the point, through mouse 
def selectPoint(event, x, y, flags, params):

    global point, condition,  old_points
    if event == cv.EVENT_LBUTTONDOWN:
        point = (int(x), int(y))
        # print(point)
        condition = True
        old_points = np.array([[x, y]], dtype=np.float32)

cv.namedWindow('frame')
cv.setMouseCallback("frame", selectPoint)
cap = cv.VideoCapture(1)
condition = False
point = ()
old_points = np.array([[]])

frame_counter = 0
starting_time = time.time()

while True:
    frame_counter +=1

    ret, frame = cap.read()
    
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #print(old_points.astype(int))
    
    if condition is True:
        cv.circle(frame, point, 5, (155, 0, 255), -1)

    # calculating frame of Video 
    fps = frame_counter/(time.time()-starting_time)
    AiPhile.textBGoutline(frame, f'FPS: {round(fps,1)}', (30, 40))

    cv.imshow('frame', frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        break
cv.destroyAllWindows()
cap.release()