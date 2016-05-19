import numpy as np
import cv2
import sys

def crop_image(name, thicken =False):
    image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2GRAY)
    raw, col =image.shape
    stride =20
    x1=x2=y1=y2=0
    for x in range(stride*3,raw, stride):
        for y in range(0,col,stride):
            if image[x][y] <100:
                x1 =x
                break
        else: continue
        break

    for y in range(stride*3,col, stride):
        for x in range(0,raw,stride):
            if image[x][y] <100:
                y1 =y
                break
        else: continue
        break

    for x in range(raw-stride*3,stride, -stride):
        for y in range(0,col,stride):
            if image[x][y] <100:
                x2 =x
                break
        else: continue
        break

    for y in range(col-stride*3,stride,-stride):
        for x in range(0,raw,stride):
            if image[x][y] <100:
                y2 =y
                break
        else: continue
        break
    image =image[x1-3*stride:x2+3*stride, y1-3*stride:y2+3*stride]  
    cv2.imwrite(name, image)


def main(argv):
    crop_image(argv)
    print u'crop success!'
    
if __name__ == "__main__":
	main(sys.argv[1])
