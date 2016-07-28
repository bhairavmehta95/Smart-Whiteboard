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
from PIL import ImageEnhance, Image
import datetime
from time import strftime
import os
from reportlab.platypus.flowables import KeepTogether
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Image


def get_image():
 retval, im = cap.read()
 return im 


def generate_pdf():
	from glob import glob
	parts = []
	time_now = datetime.datetime.now()
	#time_now = strftime("%Y-%m-%d", time_now)
	title = "generated_" + '1' + ".pdf"
	doc = SimpleDocTemplate(title, pagesize = letter)
	for file in glob('img_test/rect_*.jpg'):
		print
		parts.append(KeepTogether(Image(file)))
	doc.build(parts)

if __name__ == '__main__':
	from glob import glob
	try: src = sys.argv[1]
	except: src = 1

	ramp_frames = 30 
	cap = cv2.VideoCapture(0)
	cap.set(cv2.cv.CV_CAP_PROP_EXPOSURE, .5)
	cap.set(cv2.cv.CV_CAP_PROP_CONTRAST, 10000) 
	for i in xrange(ramp_frames):
		temp = cap.read()

	camera_capture = get_image()
	filename = "image.jpg"
	cv2.imwrite(filename, camera_capture)

	ret, img = cap.read()
	img_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	# equalize the histogram of the Y channel
	img_full = cv2.equalizeHist(img_full)
	img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

	cv2.imshow('full histogram equilization', img_full )
	cv2.imshow('Y channel equalization', img_yuv)
	cv2.imshow('frame', img)
	ch = 0xFF & cv2.waitKey()
	cv2.destroyAllWindows()
		# convert the YUV image back to RGB format
	img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

	# runs algorithm on both

	# TO DO: find a way to combine
	squares = find_squares(img)
	# squares2 = find_squares(img_output)
	# squares3 = find_squares(img_full)
	# cv2.drawContours( img, squares2, -1, (0, 255, 0), 3 )
	cv2.drawContours(img, squares, -1, (255, 0, 0), 3)
	# cv2.drawContours(img, squares3, -1, (0, 0, 255), 3)
	# cv2.drawContours(img_output, squares, -1, (255, 0, 0), 3)
	# cv2.drawContours(img_full, squares3, -1, (0, 0, 255), 3)
	cv2.imshow('squares', img)
	# cv2.imshow('squares_yuv', img_output)
	# cv2.imshow('squares_regHist', img_full)
	#generate_pdf()

	ch = 0xFF & cv2.waitKey()
	cv2.destroyAllWindows()

