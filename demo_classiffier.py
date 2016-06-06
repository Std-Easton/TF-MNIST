import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import tensorflow as tf
import cv2
from PIL import Image, ImageFilter
import sys
import types

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides =[1,1,1,1], padding ='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides =[1,2,2,1], padding ='SAME')

# Define the model (same as when creating the model file)
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')   

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

init_op = tf.initialize_all_variables()
saver = tf.train.Saver()


def predictint(imvalue, model_name ="model2.ckpt"):
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, model_name)
        #print ("Model restored.")

        return y_conv.eval(feed_dict={x: [imvalue],keep_prob: 1.0}, session=sess)[0]
    
def imageprepare(name):
    if type(name) is type(np.array([1])):
        im = Image.fromarray(name)
    #elif type(name) is type('num_9.png'):
    else:
        im = Image.open(name).convert('L')
#    im = Image.open(name).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255)) #creates white canvas of 28x28 pixels
    
    if width > height: #check which dimension is bigger
        #Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0/width*height),0)) #resize height according to ratio width
        # resize and sharpen
        img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight)/2),0)) #caculate horizontal pozition
        newImage.paste(img, (4, wtop)) #paste resized image on white canvas
    else:
        #Height is bigger. Heigth becomes 20 pixels. 
        nwidth = int(round((20.0/height*width),0)) #resize width according to ratio height
         # resize and sharpen
        img = im.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth)/2),0)) #caculate vertical pozition
        newImage.paste(img, (wleft, 4)) #paste resized image on white canvas

    tv = list(newImage.getdata()) #get pixel values
    
    tva = np.array([ (255-x)*1.0/255.0 for x in tv] )
    return tva


# NO.1 
def transformation(image, vector):
    width =image.shape[1]
    height =image.shape[0]
    M =np.array([[1,0,vector[0]],
                 [0,1,vector[1]]], dtype =float)
    dst =cv2.warpAffine(image, M, (width, height))
    return dst

# NO.2
def thicken(image, scale):
    alpha =scale-1
    m =[transformation(image,(0,alpha)),
        transformation(image,(alpha,0)),
        transformation(image,(-alpha,0)),
        transformation(image,(0,-alpha))]
    if alpha<0:
        dst =np.min(m, axis =0)
    else:
        dst =np.max(m, axis =0)
    return dst

# NO.3
def rotate(image, angle):
  	image_center = tuple(np.array(image.shape)/2)
  	rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
  	result = cv2.warpAffine(image, rot_mat, image.shape,flags=cv2.INTER_LINEAR)
  	return result

# NO.4
def elastic_transform(image, kernel=(11,11), sigma=10, alpha=8):
    from numpy.random import ranf
    import math
    displacement_field_x = np.array([[float(2*ranf(1)-1) 
               for x in range(image.shape[0])] for y in range(image.shape[1])]) * alpha
    displacement_field_y = np.array([[float(2*ranf(1)-1) 
               for x in range(image.shape[0])] for y in range(image.shape[1])]) * alpha
    displacement_field_x = cv2.GaussianBlur(displacement_field_x, kernel, sigma)
    displacement_field_y = cv2.GaussianBlur(displacement_field_y, kernel, sigma)
    result = np.zeros(image.shape)

    for row in range(image.shape[1]):
        for col in range(image.shape[0]):
            low_ii = row + math.floor(displacement_field_x[row, col])            
            high_ii = row + math.ceil(displacement_field_x[row, col])
            math.floor(displacement_field_x[row, col])
            
            low_jj = col + math.floor(displacement_field_y[row, col])
            high_jj = col + math.ceil(displacement_field_y[row, col])

            if low_ii < 0 or low_jj < 0 or high_ii >= image.shape[1] -1 \
               or high_jj >= image.shape[0] - 1:
                continue
            
            x =displacement_field_x[row, col] - math.floor(displacement_field_x[row, col])
            y =displacement_field_y[row, col] - math.floor(displacement_field_y[row, col])
            
            B =np.array([[image[low_ii, low_jj], image[low_ii, high_jj]],
                         [image[high_ii, low_jj], image[high_ii, high_jj]]])
            A =np.array([1-x, x],dtype =float)
            C =np.array([[1-y],[y]],dtype =float)
            
            result[row, col] = np.dot(A, np.dot(B,C))
            
    return result

# NO.5
def zoom(image, scale):
    x_center, y_center =image.shape
    M =np.array([[scale,0,12-24*scale/2],
                 [0,scale,12-24*scale/2]], dtype =float)
    dst =cv2.warpAffine(image, M, image.shape)
    return dst

