import argparse
import math
import bright as bright
import numpy as np
import cv2 as cv
import imutils
from PIL import Image
from matplotlib.patches import Rectangle
from numpy import asarray
from matplotlib import pyplot as plt, colors
import matplotlib.colors as mcolors


#ORB_FEATURED+Gray!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#img = cv.imread('volt.jpg')
#gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Initiate ORB detector
#orb = cv.ORB_create()
# find the keypoints with ORB
#kp = orb.detect(img, None)
#compute the ORB
#kp, des = orb.compute(img, kp)
#draw keypoints
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
#img = cv.imread("volt.jpg")
#edge = cv.Canny(img, 230, 10)
#cv.waitKey(5)
#plt.imshow(edge), plt.show()
#cv.destroyAllWindows()

#hsv!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#img = cv.imread('volt.jpg')
#original = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#hsv = cv.cvtColor(original, cv.COLOR_BGR2HSV)
#result = np.hstack((original, hsv))
#cv.waitKey(5)
#plt.imshow(result), plt.show()
#cv.destroyAllWindows()

#mirror-right-bottom!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#src = cv.imread('volt.jpg')
#original = cv.cvtColor(src, cv.COLOR_BGR2RGB)
#res = cv.flip(original, flipCode=1) #right
#res2 = cv.flip(original, flipCode=0) #bottom
#result = np.hstack((original, res, res2))
#cv.waitKey(5)
#plt.imshow(result), plt.show()
#cv.destroyAllWindows()

#rotate_45!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#image = cv.imread("volt.jpg")
#original = cv.cvtColor(image, cv.COLOR_BGR2RGB)
#rot = imutils.rotate(original, angle=45)
#result = np.hstack((original, rot))
#plt.imshow(result), plt.show()
#cv.destroyAllWindows()

#rotate with a point!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#img = cv.imread('volt.jpg', 0)
#rows, cols = img.shape
#M = cv.getRotationMatrix2D(((cols-1)/4.0, (rows-1)/1.0), 30, 1) #calculating the point and rotate
#dst = cv.warpAffine(img, M, (cols, rows))
#result = np.hstack((img, dst))
#plt.imshow(result, cmap="gray"), plt.show()
#cv.destroyAllWindows()

#shifting the img to right!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#img = cv.imread('volt.jpg', 0)
#rows, cols = img.shape
#M = np.float32([[1, 0, 50],[0, 1, 0]]) #shift the img to right (50px), we can change it to 10, but it's hard to see the difference.
#dst = cv.warpAffine(img, M, (cols, rows))
#result = np.hstack((img, dst))
#plt.imshow(result, cmap="gray"), plt.show()
#cv.destroyAllWindows()

#brightless of img!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#img = cv.imread('volt.jpg')
#original = cv.cvtColor(img, cv.COLOR_BGR2RGB)

#def change_brightness(img, value):
#    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
#    h, s, v = cv.split(hsv)
#    v = cv.add(v, value)
#   v[v > 255] = 255
#    v[v < 0] = 0
#    final_hsv = cv.merge((h, s, v))
#    original = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
#    return original
#img1 = change_brightness(original, value=255) #increases
#img2 = change_brightness(original, value=-60) #decreases
#result = np.hstack((original, img1, img2))
#plt.imshow(result), plt.show()
#cv.destroyAllWindows()

#contrast of img!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#img = cv.imread('volt.jpg')
#original = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#lab= cv.cvtColor(original, cv.COLOR_BGR2LAB)
#l_channel, a, b = cv.split(lab)
#clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
#cl = clahe.apply(l_channel)
#limg = cv.merge((cl, a, b))
#enhanced_img = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
#result = np.hstack((original, enhanced_img))
#plt.imshow(result), plt.show()
#cv.destroyAllWindows()

#CORRECTION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#img = cv.imread('volt.jpg')
#original = cv.cvtColor(img, cv.COLOR_BGR2RGB)

#hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
#h, s, v = cv.split(hsv)
#mid = 0.5
#mean = np.mean(v)
#gamma = math.log(mid*255)/math.log(mean)
#val_gamma = np.power(v, gamma).clip(0, 255).astype(np.uint8)
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
#plt.imshow(result, cmap='gray'), plt.show()
#cv.waitKey(0)
#cv.destroyAllWindows()

#balanced white WARM!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#img = cv.imread('volt.jpg')
#RGB_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)           #orig_color
#fig, ax = plt.subplots(1, figsize=(10, 10))
#img_gw = ((RGB_img * (RGB_img.mean() / RGB_img.mean(axis=(0, 1)))).clip(0, 255).astype(int))
#result = np.hstack((RGB_img, img_gw))
#plt.imshow(result), plt.show()
#cv.waitKey(0)
#cv.destroyAllWindows()

#balansed white with a zone!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#img = cv.imread('volt.jpg')
#RGB_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#def whitepatch_balancing(image, from_row, from_column, row_width, column_width):
#    fig, ax = plt.subplots(1,2, figsize=(10, 5))
#    ax[0].imshow(image)
#    ax[0].add_patch(Rectangle((from_column, from_row), column_width,
#                              row_width, linewidth=1,
#                              edgecolor='r', facecolor='none'));
#    ax[0].set_title('Original image')
#    image_patch = image[from_row:from_row+row_width, from_column:from_column+column_width]
#    image_max = (image*1.0 / image_patch.max(axis=(0, 1))).clip(0, 1)
#    ax[1].imshow(image_max);
#    ax[1].set_title('Whitebalanced Image')
#    plt.imshow(image_max), plt.show()
#    cv.waitKey(0)
#    cv.destroyAllWindows()
#whitepatch_balancing(RGB_img, 500, 1500, 120, 120)

#balanced white COLD!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#img = cv.imread('volt.jpg')
#RGB_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#def white_balance(image):
#    result = cv.cvtColor(RGB_img, cv.COLOR_BGR2LAB)
#    avg_a = np.average(result[:, :, 1])
#    avg_b = np.average(result[:, :, 0])
#    result[:, :, 2] = result[:, :, 2] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
#    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
#    result = cv.cvtColor(result, cv.COLOR_LAB2BGR)
#    return result
#final = np.hstack((RGB_img, white_balance(img)))
#plt.imshow(final), plt.show()
#cv.waitKey(0)
#cv.destroyAllWindows()

#Binarization!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#img = cv.imread('volt.jpg', cv.IMREAD_GRAYSCALE)
#img1 = cv.resize(img, (300, 300), interpolation=cv.INTER_AREA)
#ret, binary = cv.threshold(img1, 175, 255, cv.THRESH_BINARY)
#ret, binaryinv = cv.threshold(img1, 175, 255, cv.THRESH_BINARY_INV)
#ret, trunc = cv.threshold(img1, 175, 255, cv.THRESH_TRUNC)
#ret, tozero = cv.threshold(img1, 175, 255, cv.THRESH_TOZERO)
#ret, tozeroinv = cv.threshold(img1, 175, 255, cv.THRESH_TOZERO_INV)
#result = np.hstack((binary, binaryinv, trunc, tozero, tozeroinv)) #stacking images side-by-side
#plt.imshow(result, cmap='gray'), plt.show()
#cv.waitKey(0)
#cv.destroyAllWindows()

#countres!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#img = cv.imread('volt.jpg')
#img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#set a thresh
#thresh = 80
#get threshold image
#ret, thresh_img = cv.threshold(img_grey, thresh, 150, cv.THRESH_TOZERO)
#find contours
#contours, hierarchy = cv.findContours(thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#create an empty image for contours
#img_contours = np.zeros(img_grey.shape)
#draw the contours on the empty image
#result = cv.drawContours(img_contours, contours, -1, (255, 0, 0), 2)
#ora = np.hstack((img_grey, result))
#plt.imshow(ora, cmap='gray'), plt.show()
#cv.waitKey(0)
#cv.destroyAllWindows()

#Sobel-countres!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#img = cv.imread('volt.jpg')
#img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#x = cv.Sobel(img_gray, cv.CV_64F, 1, 0, ksize=3, scale=1)
#y = cv.Sobel(img_gray, -1, 1, 0, ksize=3, scale=1)
#absx= cv.convertScaleAbs(x)
#absy = cv.convertScaleAbs(y)
#edge = cv.addWeighted(absx, 0.5, absy, 0.5, 0)
#thresh = 20
#ret, thresh_img = cv.threshold(edge, thresh, 150, cv.THRESH_TOZERO)
#contours, hierarchy = cv.findContours(thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#empty = np.zeros(img_gray.shape)
#draw_c = cv.drawContours(empty, contours, -1, (255, 0, 0), 1)
#result = np.hstack((edge, draw_c))
#plt.imshow(result, cmap='gray'), plt.show()
#cv.waitKey(0)
#cv.destroyAllWindows()

#blur!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#img = cv.imread('volt.jpg')
#RGB_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)                         #orig_color
#ksize = (15, 15)
# Using cv2.blur() method
#bluray = cv.blur(RGB_img, ksize)
#result = np.hstack((RGB_img, bluray))
#plt.imshow(result), plt.show()
#cv.waitKey(0)
#cv.destroyAllWindows()

#Furfur filtrific heigh!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#img = cv.imread('volt.jpg', cv.IMREAD_GRAYSCALE)
#create mask
#r, c = img.shape
#r, c = int(r/2), int(c/2)
#Furfur
#f = np.fft.fft2(img)
#shift = np.fft.ifftshift(f)
#shift[r-30:r+30, c-30:c+30] = 0
#rev Furfur
#rev_img = np.fft.ifft2(shift)
#rev_img = np.abs(rev_img)
#result = np.hstack((img, rev_img))
#plt.imshow(result, cmap='gray'), plt.show()
#cv.waitKey(0)
#cv.destroyAllWindows()

#Furfur filtrific low!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#img = cv.imread('volt.jpg', cv.IMREAD_GRAYSCALE)
#create mask
#mask = np.zeros_like(img, dtype=np.uint8)
#r, c = mask.shape
#r, c = int(r/2), int(c/2)
#mask[r-30:r+30, c-30:c+3] = 1
#Furfur
#f = np.fft.fft2(img)
#shift = np.fft.ifftshift(f)
#shift = shift*mask
#rev Furfur
#rev_img = np.fft.ifft2(shift)
#rev_img = np.abs(rev_img)
#result = np.hstack((img, rev_img))
#plt.imshow(result, cmap='gray'), plt.show()
#cv.waitKey(0)
#cv.destroyAllWindows()

#Eroding of an img!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#img = cv.imread('volt.jpg', 0)
#RGB_img = cv.cvtColor(img, cv.COLOR_BGR2RGB) #if u vant check the rgb variant - use that and rewrite func. image and result.
#kernel = np.ones((6, 6), np.uint8)
#image = cv.erode(img, kernel, cv.BORDER_REFLECT)
# Using cv2.erode() method
#result = np.hstack((img, image))
#plt.imshow(result, cmap='gray'), plt.show()
#cv.waitKey(0)
#cv.destroyAllWindows()

#Eroding and Delation of an img!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#img = cv.imread('volt.jpg', 0)
#RGB_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# Taking a matrix of size 5 as the kernel
#kernel = np.ones((6, 6), np.uint8)
# of iterations, which will determine how much
#img_erode = cv.erode(img, kernel, iterations=1)
#img_dilation = cv.dilate(img, kernel, iterations=1)
#result = np.hstack(( img_erode, img_dilation))
#plt.imshow(result, cmap='gray'), plt.show()
#cv.waitKey(0)
#cv.destroyAllWindows()

#change color palete!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#img = cv.imread('volt.jpg')
#RGB_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#hsv_img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
#r, g, b=cv.split(RGB_img)
#img2 = cv.merge((g, r, b))
#result = np.hstack((RGB_img, img2))
#plt.imshow(result), plt.show()
#cv.waitKey(0)
#cv.destroyAllWindows()

#change color palete2!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#img = cv.imread('volt.jpg')
#RGB_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#temp = RGB_img[:,:,1].copy()
#RGB_img[:0,:,1] = RGB_img[:1,:,0]
#RGB_img[:,:,0] = temp
#plt.imshow(RGB_img), plt.show()
#cv.waitKey(0)
#cv.destroyAllWindows()

#camera!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#cap = cv.VideoCapture(0)
#while(1):
#    ret, frame = cap.read()
#    cv.imshow('Camera', frame)
#    if cv.waitKey(1) & 0xFF == ord('q'):
#         break

#cap.release()
#cv.destroyAllWindows()

#camera + sift/ORB!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#sift = cv.SIFT_create()
#orb = cv.ORB_create()
#cap = cv.VideoCapture(0)

#while cap.isOpened():

#    ret, img1 = cap.read()

#    img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
#    kp, descriptors_1 = orb.detectAndCompute(img1, None)
#    img2 = cv.drawKeypoints(img1, kp, cv.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG, flags=0)

#    cv.imshow('SIFT', img2)

#    if cv.waitKey(5) & 0xFF == 27:
#        break

#cap.release()
#cv.destroyAllWindows()
