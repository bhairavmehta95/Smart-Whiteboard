import numpy as np
import cv2
import cv2.cv as cv
from PIL import ImageChops
import PIL.Image
#from skimage.measure import compare_ssim as ssim
import math, operator

# class to hold the center points of an image, used to make sure we don't capture the same image twice
class Center:
    def __init__(self, x_, y_):
        self.x = x_
        self.y = y_

# function to calculate distance between two center point objects
def distance_centers(test_point, center_point):
    x_sq = (test_point.x - center_point.x) ** 2
    y_sq = (test_point.y - center_point.y) ** 2
    return (x_sq + y_sq) ** .5

# checks if the center points are less than 30 pixels away from each other
def is_similar(test_point, centers):
    for center_point in centers:
        distance = distance_centers(test_point, center_point)
        if distance < 30:
            return True

    return False

# uses root mean square calculations to see how similar two images are
# TO DO: Remimplement (currently not utilized)
# def rms_evaluator():
#     roi_im = cv2.imread('fakepath.jpg')

#     roi_im = cv2.cvtColor(roi_im, cv2.COLOR_BGR2GRAY)

#     from glob import glob
#     for fn in glob('img_test/*.jpg'):
#         saved = cv2.imread(fn)
#         saved = cv2.cvtColor(saved, cv2.COLOR_BGR2GRAY)
#         if abs(len(roi_im) - len(saved)) > .1*(min(len(roi_im), len(saved))):
#             return False
#         else:
#             min_length = 0
#             print roi_im.shape, saved.shape
#             if len(roi_im) > len(saved):
#                 # CROP ROI
#                 print "Cropping ROI"
#                 roi_im = roi_im[0:saved.shape[0], 0:saved.shape[1]]
#                 min_length = len(saved)

#                 if saved.shape[0] > roi_im.shape[0]:
#                     saved = saved[0:roi_im.shape[0], 0:saved.shape[1]]
#                 if saved.shape[1] > roi_im.shape[1]:
#                     saved = saved[0:saved.shape[0], 0:roi_im.shape[1]]
#             else:
#                 # CROP SAVED
#                 print "Cropping Saved"
#                 saved = saved[0:roi_im.shape[0], 0:roi_im.shape[1]]
#                 min_length = len(roi_im)

#                 if roi_im.shape[0] > saved.shape[0]:
#                     roi_im = roi_im[0:saved.shape[0], 0:roi_im.shape[1]]
#                 if roi_im.shape[1] > saved.shape[1]:
#                     roi_im = roi_im[0:roi_im.shape[0], 0:saved.shape[1]]
#             try:
#                 ssim_out = ssim(roi_im, saved)
#                 print "SSIM:", ssim_out
#                 if ssim_out > .5:
#                     return True
#                 else:
#                     return False
#             except:
#                 print "SSIM FAILED: ", roi_im.shape, saved.shape
#                 return False

# calculates cosines between three points
def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

# function to find all of the squares within an image
def find_squares(save_count):
    img = cv2.imread('fakepath.jpg')
    squares = []
    centers = []

    # count = 0
    # sigma=0.33
    # # compute the median of the single channel pixel intensities
    # v = np.median(img)
 
    # # apply automatic Canny edge detection using the computed median
    # lower = int(max(0, (1.0 - sigma) * v))
    # upper = int(min(255, (1.0 + sigma) * v))

    for gray in cv2.split(img):
        # thresholding, dilation
        for thrs in xrange(0, 255, 255): ## CHANGE TO 255 WHEN DOING image differences, 26 OTHERWISE
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                #bin = cv2.Canny(gray, lower, upper)
                bin = cv2.dilate(bin, None)
            else:
                retval, bin = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # loops through the contours found
            for cnt in contours:
                # approximates the polynomial
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.05*cnt_len, True)
                
                # the shape has 4 sides so it is some sort of rectangle/square
                if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
                    (x, y, w, h) = cv2.boundingRect(cnt)
                    roi = cv2.boundingRect(cnt)
                    cnt = cnt.reshape(-1, 2)
                    
                    # finds the max cosine
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < .4:
                        ar = w / float(h)
                        # a square will have an aspect ratio that is approximately
                        # equal to one, otherwise, the shape is a rectangle
                        shape = "square" if ar >= 0.97 and ar <= 1.03 else "rectangle"  
                        string = "whiteboard_session/squares/" + shape + str(save_count) + ".jpg"  
                        roi_im = img[y:y+h, x:x+w]
                        cv2.imwrite('fakepath.jpg', roi_im)                                          

                        im = PIL.Image.open('fakepath.jpg')

                        # will not be neccesary with smaller images
                        colors = im.getcolors(im.size[0] * im.size[1])

                        values = colors[0][1]

                        value_max = max(values[0], values[1], values[2])
                        value_min = min(values[0], values[1], values[2])

                        if abs(value_max - value_min) < 30:
                            continue

                        x_c = (2 * x + w) / 2
                        y_c = (2 * y + h) / 2
                        center_point = Center(x_c, y_c)

                        similarity = is_similar(center_point, centers)
                        # rms_eval = rms_evaluator()

                        if not similarity: #and not rms_eval:
                            print "writing", shape, " with id: ", string
                            centers.append(center_point)
                            cv2.imwrite(string, roi_im)
                            save_count = save_count + 1

                        squares.append(cnt) 
                       # print shape

    img_copy = img
    cv2.drawContours( img_copy, squares, -1, (255, 255, 255), 3 )
    try:
        cv2.imshow('squares', img_copy)
        ch = 0xFF & cv2.waitKey()
    except:
        print "cannot show image, returning squares"
    
    return squares
