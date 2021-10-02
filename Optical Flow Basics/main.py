import cv2 as cv
import numpy as np
import AiPhile
import time

def selectPoint(event, x, y, flags, params):
    global point, condition,  old_points
    if event == cv.EVENT_LBUTTONDOWN:
        point = (int(x), int(y))
        # print(point)
        condition = True
        old_points = np.array([[x, y]], dtype=np.float32)

lk_params = dict(winSize=(20, 20),
                 maxLevel=4,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.01))
  
        
cap = cv.VideoCapture(0)
_, frame = cap.read()
old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

cv.namedWindow('frame')
cv.setMouseCallback("frame", selectPoint)

condition = False
point = ()
old_points = np.array([[]])

frame_counter = 0
starting_time = time.time()

while True:
    frame_counter +=1
    ret, frame = cap.read()
    cv.imshow('old frame ', old_gray)
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #print(old_points.astype(int))
    if condition is True:
        cv.circle(frame, point, 5, (155, 0, 255), -1)
        new_points, status, error = cv.calcOpticalFlowPyrLK(
            old_gray, gray_frame, old_points, None, **lk_params)
        old_points = new_points
        new_points=new_points.astype(int)
        #print(type(new_points))
        x, y = new_points.ravel()
        cv.line(frame, (x+2,y-2) ,(x+20, y-30), AiPhile.GREEN, 2, cv.LINE_AA)
        AiPhile.textBGoutline(frame, f'Tracking Point', (x+20, y-30),scaling=0.5)
        
        #print(x, y)
        
        cv.circle(frame, (x, y), 6, (0, 255, 255), 4)
    fps = frame_counter/(time.time()-starting_time)
    AiPhile.textBGoutline(frame, f'FPS: {round(fps,1)}', (30, 40))
    old_gray = gray_frame.copy()
    cv.imshow('frame', frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
cv.destroyAllWindows()
cap.release()
