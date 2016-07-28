#!/usr/bin/env python

import sys

sys.path.append('/usr/local/lib/python2.7/site-packages')

import numpy
import cv2
import cv2.cv as cv
from squares import *
import os
import time
import sys
import video
from common import mosaic
from PIL import ImageEnhance, Image,ImageChops
from skimage.measure import compare_ssim as ssim
import math, operator
import datetime
from time import strftime
import os
from reportlab.platypus.flowables import KeepTogether
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Image

# Global variables for pdf
parts = []
doc = SimpleDocTemplate('generated.pdf', pagesize = letter)

# TO DO: Fix this so it takes in a numpy array
def compute_dominant_pixel(): 
	im = Image.open('image.jpg')

	# will not be neccesary with smaller images
	colors = im.getcolors(im.size[0] * im.size[1])

	print colors[0]

def get_image():
 retval, im = cap.read()
 return im 


def generate_pdf():
	from glob import glob
	global parts, doc

	# TO DO: FIX THIS so it only adds certain images
	for file in glob('whiteboard_sesson/*.jpg'):
		parts.append(KeepTogether(Image(file)))
	

def start_up():
	try: src = sys.argv[1]
	except: src = 1

	ramp_frames = 30 
	cap = cv2.VideoCapture(0)
	cap.set(cv2.cv.CV_CAP_PROP_EXPOSURE, .5)
	cap.set(cv2.cv.CV_CAP_PROP_CONTRAST, 10000) 
	for i in xrange(ramp_frames):
		temp = cap.read()

	# capture, save image
	camera_capture = get_image()
	filename = "whiteboard_session/prev.jpg"
	cv2.imwrite(filename, camera_capture)

	return cap

class Center:
    def __init__(self, x_, y_):
        self.x = x_
        self.y = y_

def distance_centers(test_point, center_point):
    x_sq = (test_point.x - center_point.x) ** 2
    y_sq = (test_point.y - center_point.y) ** 2
    return (x_sq + y_sq) ** .5

def is_similar(test_point, centers):
    for center_point in centers:
        distance = distance_centers(test_point, center_point)
        if distance < 30:
            return True

    return False

def rms_evaluator():
    roi_im = cv2.imread('fakepath.jpg')

    roi_im = cv2.cvtColor(roi_im, cv2.COLOR_BGR2GRAY)

    from glob import glob
    for fn in glob('img_test/*.jpg'):
        saved = cv2.imread(fn)
        saved = cv2.cvtColor(saved, cv2.COLOR_BGR2GRAY)
        if abs(len(roi_im) - len(saved)) > .1*(min(len(roi_im), len(saved))):
            return False
        else:
            min_length = 0
            print roi_im.shape, saved.shape
            if len(roi_im) > len(saved):
                # CROP ROI
                print "Cropping ROI"
                roi_im = roi_im[0:saved.shape[0], 0:saved.shape[1]]
                min_length = len(saved)

                if saved.shape[0] > roi_im.shape[0]:
                    saved = saved[0:roi_im.shape[0], 0:saved.shape[1]]
                if saved.shape[1] > roi_im.shape[1]:
                    saved = saved[0:saved.shape[0], 0:roi_im.shape[1]]
            else:
                # CROP SAVED
                print "Cropping Saved"
                saved = saved[0:roi_im.shape[0], 0:roi_im.shape[1]]
                min_length = len(roi_im)

                if roi_im.shape[0] > saved.shape[0]:
                    roi_im = roi_im[0:saved.shape[0], 0:roi_im.shape[1]]
                if roi_im.shape[1] > saved.shape[1]:
                    roi_im = roi_im[0:roi_im.shape[0], 0:saved.shape[1]]
            try:
                ssim_out = ssim(roi_im, saved)
                print "SSIM:", ssim_out
                if ssim_out > .5:
                    return True
                else:
                    return False
            except:
                print "SSIM FAILED: ", roi_im.shape, saved.shape
                return False

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def take_picture():
	ret, img = cap.read()

	original = cv2.imread('whiteboard_session/prev.jpg')
	changed = img

	# converts, blurs images for image processing
	img_a = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
	img_a = cv2.GaussianBlur(img_a, (11, 11), 0)

	img_b = cv2.cvtColor(changed, cv2.COLOR_BGR2GRAY)
	img_b = cv2.GaussianBlur(img_b, (11, 11), 0)

	img_diff = cv2.absdiff(img_a, img_b)

	thresh = cv2.threshold(img_diff, 15, 255, cv2.THRESH_BINARY)[1]

	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)
	(contours, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)

	contours_final = []

	# loop over the contours
	for c in contours:

		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		rect = cv2.rectangle(img_diff, (x, y), (x + w, y + h), (255, 0, 0), 2)
		
		cnt_len = cv2.arcLength(c, True)
		cnt = cv2.approxPolyDP(c, 0.05*cnt_len, True)

		shape = 'default'

		# is it a square, rectangle, or other
		if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
			(x, y, w, h) = cv2.boundingRect(cnt)
			roi = cv2.boundingRect(cnt)
			cnt = cnt.reshape(-1, 2)

			max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])

			if max_cos < .4:
				ar = w / float(h)
				# a square will have an aspect ratio that is approximately
				# equal to one, otherwise, the shape is a rectangle
				shape = "square" if ar >= 0.97 and ar <= 1.03 else "rectangle" 

				string = "img_test/" + shape + str(count) + ".jpg"  
				roi_im = img[y:y+h, x:x+w]
				cv2.imwrite('fakepath.jpg', roi_im)
				x_c = (2 * x + w) / 2
				y_c = (2 * y + h) / 2
				center_point = Center(x_c, y_c)

				similarity = is_similar(center_point, centers)
				rms_eval = rms_evaluator() 

				x_c = (2 * x + w) / 2
				y_c = (2 * y + h) / 2
				center_point = Center(x_c, y_c)

				similarity = is_similar(center_point, centers)
				rms_eval = rms_evaluator()

				if not similarity and not rms_eval:
					print "writing", shape, " with id: ", string
					centers.append(center_point)
					cv2.imwrite(string, roi_im)
					count = count + 1


		# TO DO: add shape information with this
		contours_final.append(rect)

	cv2.drawContours(original, contours_final, -1, (255, 0, 0), 3 )
	cv2.imshow('im',original)
	cv2.imshow('im removed', changed)

	ch = 0xFF & cv2.waitKey()
	cv2.destroyAllWindows()

	cv2.imshow('imdf',img_diff)

	ch = 0xFF & cv2.waitKey()
	cv2.destroyAllWindows()
	


if __name__ == '__main__':
	from glob import glob
	global doc
	cap = start_up()

	while True:
		val = raw_input("Please type h(elp), p(icture), or q(uit)")
		if val == 'h':
			print "This is a program that does fun things, try another command!"
		elif val == 'q':
			doc.build(parts)
			break
		elif val == 'p':
			take_picture(cap)
		else:
			print "Not a valid command, try again"

	