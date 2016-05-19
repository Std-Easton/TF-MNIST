import numpy as np
import cv2
import sys


# NO.1 
def transformation(image, vector):
    width =image.shape[1]
    height =image.shape[0]
    M =np.array([[1,0,vector[0]],
                 [0,1,vector[1]]], dtype =float)
    dst =cv2.warpAffine(image, M, (width, height))
    return dst

# NO.2
def thickness(image, scale):
    alpha =scale-1
    m =[transformation(image,(0,alpha)),
        transformation(image,(alpha,0)),
        transformation(image,(-alpha,0)),
        transformation(image,(0,-alpha))]
    if alpha>0:
        dst =np.min(m, axis =0)
    else:
        dst =np.max(m, axis =0)
    edge =int(alpha)+1
	det =dst[edge:width-edge,edge:height-edge]
    return dst

def main(argv):
    name =argv
    image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2GRAY)
    image =thickness(image,3)
    cv2.imwrite(name, image)
    print u'thicken success!'
    
if __name__ == "__main__":
	main(sys.argv[1])
