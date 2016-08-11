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
from PIL import ImageEnhance, ImageChops
import PIL.Image
from skimage.measure import compare_ssim as ssim
import math, operator
import datetime
from time import strftime
import os

# PDF Import statements
from reportlab.platypus.flowables import KeepTogether
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, Spacer

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.rl_config import defaultPageSize
from reportlab.lib.units import inch

# Whiteboard code imports
from pits_whiteboard import find_squares
import speech_recognition as sr
import threading
from gen_wordcloud import generate_wordcloud
import generate_gif


# Global variables for pdf
styles = getSampleStyleSheet()
parts = []
doc = SimpleDocTemplate('generated.pdf', pagesize = letter)
save_count = 0

# Boolean variables for control floq
is_taking_picture = False
camera_is_dead = False

# string to store full transcript (Needs to be recorded in speech thread, used in whiteboard thread)
full_transcript = ''

# A function that asks the user to speak and records what he/she said during the recording
# Uses google cloud speech API through SpeechRecognition Python package
def speech_query():
	r = sr.Recognizer()
	m = sr.Microphone()

	try:
		print("A moment of silence, please...")
		with m as source: r.adjust_for_ambient_noise(source)
		print("Set minimum energy threshold to {}".format(r.energy_threshold))

		print("Say something!")
		with m as source: audio = r.listen(source)
		print("Got it! Now to recognize it...")
		try:
			# recognize speech using Google Speech Recognition
			value = r.recognize_google(audio)

			# we need some special handling here to correctly print unicode characters to standard output
			if str is bytes: # this version of Python uses bytes for strings (Python 2)
				print(u"You said {}".format(value).encode("utf-8"))
			else: # this version of Python uses unicode for strings (Python 3+)
				print("You said {}".format(value))

			return value
		except sr.UnknownValueError:
			print("Oops! Didn't catch that")
		except sr.RequestError as e:
			print("Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))
	except KeyboardInterrupt:
		pass


# TO DO: Fix this so it takes in a numpy array
# A function that computes the most dominant color and the number of occurances
def compute_dominant_pixel(): 
	im = PIL.Image.open('fakepath.jpg')
	# will not be neccesary with smaller images
	colors = im.getcolors(im.size[0] * im.size[1])
	R = 0
	G = 0
	B = 0
	count = 0

	for color in colors:
		# weighted average for most dominant pixel
		R += color[1][0] * color[0]
		G += color[1][1] * color[0]
		B += color[1][2] * color[0]
		count += color[0]

	print R/count, G/count, B/count
	return [R/count, G/count, B/count]



# A function that simply reads and returns the image from 'cap(ture device)'
def get_image(cap):
 retval, im = cap.read()
 return im 

# Builds the final PDF from global parts list
def generate_pdf():
	global parts

	doc.build(parts)

# helper to checks if color value is APPROXIMATELY white
# if numbers are approx the same, then they are white/gray
def dominant_pixel_diff(values):
	min_val = min(values[0], values[1], values[2])
	max_val = max(values[0], values[1], values[2])

	# is approximately white if min and max difference is close
	return abs(min_val - max_val)


# A function that sets up the camera and takes the original bg picture
def start_up():
	global BACKGROUND

	try: 
		src = 1
		cap = cv2.VideoCapture(src)

	except: 
		src = 0
		cap = cv2.VideoCapture(src)

	ramp_frames = 30 
	cap = cv2.VideoCapture(0)
	#cap = cv2.VideoCapture(src)

	# cap.set(cv2.cv.CV_CAP_PROP_EXPOSURE, .5)
	# cap.set(cv2.cv.CV_CAP_PROP_CONTRAST, 10000) 

	for i in xrange(ramp_frames):
		temp = cap.read()


	# capture, save image
	camera_capture = get_image(cap)
	filename = "whiteboard_session/full/0.jpg"

	cv2.imwrite(filename, camera_capture)

	return cap

# A function that calculates cosines between edges
def angle_cos(p0, p1, p2):
	d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
	return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

# A function that takes, analyzes, and outputs a picture (to a PDF)
def take_picture(cap, pic_count, centers):
	ret, img = cap.read()
	ret, img = cap.read()

	global save_count

	# so we can save at the right spot
	start_save_count = save_count

	# uses last picture as BG, saves this picture as new (changed)
	original_str = 'whiteboard_session/full/' + str(pic_count - 1) + '.jpg'
	changed_str = 'whiteboard_session/full/' + str(pic_count) + '.jpg'

	cv2.imwrite(changed_str, img)

	original = cv2.imread(original_str)
	changed = img

	# Show the original
	cv2.imshow('Background Image',original)
	ch = 0xFF & cv2.waitKey()
	cv2.destroyAllWindows()

	img_changes = changed

	# converts, blurs images for image processing
	img_a = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
	img_a = cv2.GaussianBlur(img_a, (5, 5), 0)

	img_b = cv2.cvtColor(changed, cv2.COLOR_BGR2GRAY)
	img_b = cv2.GaussianBlur(img_b, (5, 5), 0)

	# calculates difference, thresholds
	img_diff = cv2.absdiff(img_a, img_b)

	ret, thresh = cv2.threshold(img_diff,25,255,cv2.THRESH_BINARY)


	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	cv2.imshow('Thresholded', thresh) 

	ch = 0xFF & cv2.waitKey()
	cv2.destroyAllWindows()

	thresh = cv2.dilate(thresh, None, iterations=2)
	#thresh = cv2.dilate(thresh, None)
	
	cv2.imshow('Image Difference', img_diff)
	cv2.imshow('Dilated', thresh) 

	ch = 0xFF & cv2.waitKey()
	cv2.destroyAllWindows()

	#(contours, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		#cv2.CHAIN_APPROX_SIMPLE)
	(contours, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.RETR_LIST)


	# lists to store the dictionaries of contours
	contours_final = []
	context_dict = []
	count = 0

	# loop over the contours
	for c in contours:
		context = {}

		# compute the bounding box for the contour
		(x, y, w, h) = cv2.boundingRect(c)

		# draw it on frame
		#rect = cv2.rectangle(img_changes, (x, y), (x + w, y + h))
		
		# if the area is too small, skip it
		if cv2.contourArea(c) < 1000:
			continue

		# AR = w:h
		aspect_ratio = w/float(h)

		# too lopsided, probably not the image we want
		if aspect_ratio >= 8.0 or aspect_ratio <= .125:
			continue

		# Do we need this here?? :: I dont't think so TODO
		cnt_len = cv2.arcLength(c, True)
		cnt = cv2.approxPolyDP(c, 0.05*cnt_len, True)

		shape = 'default'
		
		# Makes a bounding rectangle
		(x, y, w, h) = cv2.boundingRect(cnt)
		roi = cv2.boundingRect(cnt)
		cnt = cnt.reshape(-1, 2)

		# check if post it note

		############# POST ITS

		# splices the image where the difference was found (gives it more than just what was there)
		roi_original = original[y:y+h, x:x+w]
		roi_changed = changed[y:y+h, x:x+w]

		try:
			cv2.imshow('original', roi_original)
			cv2.imshow('changed', roi_changed)
			ch = 0xFF & cv2.waitKey()
			cv2.destroyAllWindows()
		except:
			pass

		# computes dominant color and pixel
		cv2.imwrite('fakepath.jpg', roi_original)

		original_dominant = compute_dominant_pixel()

		cv2.imwrite('fakepath.jpg', roi_changed)	

		changed_dominant = 	compute_dominant_pixel()

		original_dominant = dominant_pixel_diff(original_dominant)
		changed_dominant = dominant_pixel_diff(changed_dominant)

		# Saves the changed file
		save_string = 'whiteboard_session/img_test/' + str(save_count) + '.jpg'
		save_count += 1

		original_is_homogenous = False
		changed_is_homogenous = False

		# very close
		if abs(original_dominant - changed_dominant) < 30:			
			cv2.imwrite('fakepath.jpg', roi_original)
			img_a = cv2.imread('fakepath.jpg')
			img_a = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
			img_a = cv2.GaussianBlur(img_a, (5, 5), 0)

			# bin = cv2.Canny(gray, 0, 50, apertureSize=5)
			# #bin = cv2.Canny(gray, lower, upper)
			# bin = cv2.dilate(bin, None)

			retval, a = cv2.threshold(img_a, 10, 255, cv2.THRESH_BINARY)
			contours_original, hierarchy = cv2.findContours(a, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

			cv2.imwrite('fakepath.jpg', roi_changed)
			img_b = cv2.imread('fakepath.jpg')
			img_b = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
			img_b = cv2.GaussianBlur(img_b, (5, 5), 0)

			# bin = cv2.Canny(gray, 0, 50, apertureSize=5)
			# #bin = cv2.Canny(gray, lower, upper)
			# bin = cv2.dilate(bin, None)

			retval, b = cv2.threshold(img_b, 10, 255, cv2.THRESH_BINARY)
			contours_changed, hierarchy = cv2.findContours(b, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

			print "len(original):", len(contours_original), "& len (changed):", len(contours_changed) 
			if len(contours_original) <= len(contours_changed):
				original_is_homogenous = True


		# Original is more homogenous, so it is closer to a white or gray
		if changed_dominant > original_dominant or original_is_homogenous:
			context['change'] = 'added'
			try:
				roi_changed = changed[y-h/4:y+(5/4)*h, x-w/4:x+(5/4)*w]
			except:
				pass
			cv2.imwrite('fakepath.jpg', roi_changed)

			#img = cv2.imread('fakepath.jpg')
			#cv2.imshow('img', img)

			ch = 0xFF & cv2.waitKey()
			cv2.destroyAllWindows()


			count_ = save_count
			find_squares(count_)
			cv2.imwrite(save_string, roi_changed)

		else:
			context['change'] = 'removed'
			try:
				roi_original = original[y-h/4:y+(5/4)*h, x-w/4:x+(5/4)*w]
			except:
				pass
			cv2.imwrite('fakepath.jpg', roi_original)

			#img = cv2.imread('fakepath.jpg')
			#cv2.imshow('img', img)

			ch = 0xFF & cv2.waitKey()
			cv2.destroyAllWindows()

			count_ = save_count
			find_squares(count_)
			cv2.imwrite(save_string, roi_original)
		

		context['shape'] = shape

		# Adds them to the list
		context_dict.append(context)

	# Stores where we end the save count
	end_save_count = save_count

	string_to_write = 'whiteboard_session/changes/' + str(pic_count) + '.jpg'
	cv2.imwrite(string_to_write, img_changes)
	
	#### BUILDING THE PDF ####

	# build into document
	global parts
	style = styles["Normal"]
	title = 'Start of timestamp: ' + str(start_save_count)
	p = Paragraph(title, style)
 	parts.append(p)
 	string = 'Full board at timestamp:' + str(start_save_count)
 	file = 'whiteboard_session/full/' + str(pic_count) + '.jpg'
 	p = Paragraph(string, style)
 	parts.append(p)
 	parts.append(KeepTogether(Image(file)))


 	# Added post its
 	
 	tmp = start_save_count
 	it = 0

 	string = 'Post Its Added at timestamp:' + str(start_save_count)
 	p = Paragraph(string, style)
 	parts.append(p)

	while tmp < end_save_count:
		if context_dict[it]['shape'] == 'post' and context_dict[it]['change'] == 'added':
			file = 'whiteboard_session/img_test/' + str(start_save_count) + '.jpg'
			parts.append(KeepTogether(Image(file)))
		tmp += 1
		it += 1

	# Removed post its
 	
 	tmp = start_save_count
 	it = 0

 	string = 'Post Its Removed at timestamp:' + str(start_save_count)
 	p = Paragraph(string, style)
 	parts.append(p)

	while tmp < end_save_count:
		if context_dict[it]['shape'] == 'post' and context_dict[it]['change'] == 'removed':
			file = 'whiteboard_session/img_test/' + str(start_save_count) + '.jpg'
			parts.append(KeepTogether(Image(file)))
		tmp += 1
		it += 1

	## End generation of PDF

	#### SHOWING WHAT WAS CHANGED ####

	original_copy = original
	cv2.drawContours(original_copy, contours_final, -1, (255, 0, 0), 3 )
	save_string = 'whiteboard_session/pdf/' + str(pic_count) + '.jpg'
	cv2.imwrite(save_string, changed)
	cv2.imshow('im',original_copy)
	cv2.imshow('im removed', changed)

	ch = 0xFF & cv2.waitKey()
	cv2.destroyAllWindows()

	cv2.imshow('imdf', img_changes)

	ch = 0xFF & cv2.waitKey()
	cv2.destroyAllWindows()


	return centers

# thread class that runs the speech_query function and adds the text to the full transcript
class speech_thread(threading.Thread):
	def __init__(self, threadID, name, counter):
		threading.Thread.__init__(self)
		self.threadID = threadID
		self.name = name
		self.counter = counter
	def run(self):
		global is_taking_picture, full_transcript, camera_is_dead
		print "Starting " + self.name
		while True:
			if camera_is_dead:
				return
			if not is_taking_picture:
				text = speech_query()
				try:
					full_transcript += text
					full_transcript += '\n'
				except:
					pass
		print "Exiting " + self.name

# thread class to prompt the user with picture, quit, or help
class picture_thread(threading.Thread):
	def __init__(self, threadID, name, counter):
		threading.Thread.__init__(self)
		self.threadID = threadID
		self.name = name
		self.counter = counter
	def run(self):
		global is_taking_picture, full_transcript
		print "Starting " + self.name
		print "Starting up, please wait..."
		cap = start_up()
		count = 1
		centers = []
		while True:
			print "Please type h(elp), p(icture), or q(uit)"
			val = raw_input('> ')
			if val == 'h' or val == 'help':
				print "This is a program that does fun things, try another command!"
			elif val == 'q' or val == 'quit':
				doc.build(parts)
				try:
					file = open('full_transcript.txt', 'w')
					file.write(full_transcript)
					file.close()
					generate_wordcloud()
					generate_gif()
				except:
					pass
				break
			elif val == 'p' or val == 'picture':
				is_taking_picture = True
				print "Your transcript is: ", full_transcript
				centers = take_picture(cap, count, centers)
				count += 1
				is_taking_picture = False
			else:
				print "Not a valid command, try again"
		print "Exiting " + self.name

# main function that runs the whiteboard, spawns a picture thread and a speech thread
# then constantly checks to see if picture thread is dead (if so, kills both)
def run_whiteboard():
	from glob import glob
	# global doc

	# intialize threads, set as daemons so they exit when main thread ends
	s = speech_thread(1, "speech-thread", 1)
	s.daemon = True
	p = picture_thread(2, "picture-thread", 2)
	p.daemon = True

	# Start new Threads
	s.start()
	p.start()
	

	#Kills both threads if user presses q
	while True:
		if not p.is_alive():
			return
			

if __name__ == '__main__':
	run_whiteboard()