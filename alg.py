import math
import numpy as np
import cv2 as cv
import imutils
from PIL import Image
from numpy import asarray
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors


#ORB_FEATURED+Gray!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#img = cv.imread('volt.jpg')
#gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Initiate ORB detector
#orb = cv.ORB_create()
# find the keypoints with ORB
#kp = orb.detect(img, None)
#compute the descriptors with ORB
#kp, des = orb.compute(img, kp)
#draw only keypoints location,not size and orientation
#img = cv.drawKeypoints(gray, None, None, color=(0, 255, 0))
#img2 = cv.drawKeypoints(gray, kp, None, color=(0, 255, 0), flags=0)
#result = np.hstack((img, img2))
#cv.waitKey(5)
#plt.imshow(result), plt.show()
#cv.destroyAllWindows()

#SIFT_FEATURED+Gray!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#img = cv.imread('volt.jpg')
#gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#sift = cv.SIFT_create()
#kp = sift.detect(gray, None)
#img = cv.drawKeypoints(gray, None, None, color=(0, 255, 0))
#img1 = cv.drawKeypoints(gray, kp, None, color=(0, 255, 0), flags=0)
#result = np.hstack((img, img1))
#cv.waitKey(5)
#plt.imshow(result), plt.show()
#cv.destroyAllWindows()

#canny edges!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#img = cv.imread("volt.jpg")  # Read image
#Applying the Canny Edge filter
#edge = cv.Canny(img, 230, 10)
#cv.waitKey(5)
#plt.imshow(edge), plt.show()
#cv.destroyAllWindows()

# hsv!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#img = cv.imread('volt.jpg')
#hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
#result = np.hstack((img, hsv))
#cv.waitKey(5)
#plt.imshow(result), plt.show()
#cv.destroyAllWindows()

#mirror-right-bottom!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#src = cv.imread('volt.jpg')
#res = cv.flip(src, flipCode=1) #right
#res2 = cv.flip(src, flipCode=0) #bottom
#result = np.hstack((src, res, res2))
#cv.waitKey(5)
#plt.imshow(result), plt.show()
#cv.destroyAllWindows()

#rotate_45!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#image = cv.imread("volt.jpg")

#rot = imutils.rotate(image, angle=45)
#result = np.hstack((image, rot))
#plt.imshow(result), plt.show()
#cv.destroyAllWindows()

#rotate with a point!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#img = cv.imread('volt.jpg', 0)
#rows, cols = img.shape
#M = cv.getRotationMatrix2D(((cols-1)/4.0, (rows-1)/1.0), 30, 1) #calculating the point and rotate
#dst = cv.warpAffine(img, M, (cols, rows))
#result = np.hstack((img, dst))
#plt.imshow(result), plt.show()
#cv.destroyAllWindows()

#shifting the img to right!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#img = cv.imread('volt.jpg', 0)
#rows, cols = img.shape
#M = np.float32([[1,0,50],[0,1,0]]) #shift the img to right (50px), we can change it to 10, but it's hard to see the difference.
#dst = cv.warpAffine(img,M,(cols,rows))
#result = np.hstack((img, dst))
#plt.imshow(result), plt.show()
#cv.destroyAllWindows()

#brightless of img!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#img = cv.imread('volt.jpg')

#def change_brightness(img, value):
#    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
#    h, s, v = cv.split(hsv)
#    v = cv.add(v,value)
#    v[v > 255] = 255
#    v[v < 0] = 0
#    final_hsv = cv.merge((h, s, v))
#    img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
#    return img
#img1 = change_brightness(img, value=255) #increases
#img2 = change_brightness(img, value=-60) #decreases
#result = np.hstack((img, img1, img2))

#plt.imshow(result), plt.show()
#cv.destroyAllWindows()

#contrast of img!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#img = cv.imread('volt.jpg', 1)
#lab= cv.cvtColor(img, cv.COLOR_BGR2LAB)
#l_channel, a, b = cv.split(lab)
#clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#cl = clahe.apply(l_channel)
#limg = cv.merge((cl, a, b))
#enhanced_img = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
#result = np.hstack((img, enhanced_img))
#plt.imshow(result), plt.show()
#cv.destroyAllWindows()

#CORRECTION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#img = cv.imread('volt.jpg')
#original = cv.cvtColor(img, cv.COLOR_BGR2RGB)

#hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
#h, s, v = cv.split(hsv)

# compute gamma = log(mid*255)/log(mean)
#mid = 0.5
#mean = np.mean(v)
#gamma = math.log(mid*255)/math.log(mean)

# do gamma correction on value channel
#val_gamma = np.power(v, gamma).clip(0, 255).astype(np.uint8)

# combine new value channel with original hue and sat channels
#hsv_gamma = cv.merge([h, s, val_gamma])
#img_gamma2 = cv.cvtColor(hsv_gamma, cv.COLOR_HSV2RGB)

#result = np.hstack((original, img_gamma2))
#plt.imshow(result), plt.show()
#cv.waitKey(0)
#cv.destroyAllWindows()

#hystogramm_equalization!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#img = cv.imread('volt.jpg', 0)
#equ = cv.equalizeHist(img)
#result = np.hstack((img, equ)) #stacking images side-by-side
#plt.imshow(result), plt.show()
#cv.waitKey(0)
#cv.destroyAllWindows()

