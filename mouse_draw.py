import cv2
import numpy as np
from demo_classiffier import *
from crop_image import *
import matplotlib.pyplot as plt

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
tempx, tempy =-1,-1

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode, tempx, tempy
    thickness =28
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
        tempx, tempy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
#            cv2.circle(img,(x,y),7,(0,0,255),-1)
            cv2.line(img,(x,y),(tempx,tempy),(0,0,255),thickness)
            tempx, tempy = x,y


    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
#        cv2.circle(img,(x,y),7,(0,0,255),-1)
        cv2.line(img,(x,y),(tempx,tempy),(0,0,255),thickness)
        tempx, tempy = x,y

img = np.ones((512,512,3), np.uint8) * 255
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k is ord('q'):
        cv2.destroyAllWindows()
        break
    elif k is ord('c'):
        img =np.ones((512,512,3), np.uint8) * 255
    elif k is ord('p'):
        ## image preprocessing part
        img_processed =imageprepare(crop_image(img))
        ## predict part
        predint = predictint(img_processed, "model2-1.ckpt")
        ## showing predition part
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'predicted number: '+str(np.argmax(predint)),
                    (10,475), font, 1,(0,0,0),2)
        cv2.putText(img,'confidence rate: '+str(round(max(predint),2)),
                    (10,500), font, 1,(0,0,0),2)

