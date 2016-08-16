from images2gif_edited import writeGif
import PIL.Image as Image
import os

from PIL import Image
import numpy, cv2

import cv2.cv as cv

# def generate_gif():
# 	from glob import glob
# 	images = []
# 	for img in glob('whiteboard_session/full/*.jpg'):
# 		images.append(Image.open(img))
# 		images[-1].show()
# 	size = (600,350)

# 	for im in images:
# 	    im.thumbnail(size, Image.ANTIALIAS)

# 	print len(images)

# 	filename = "filename.gif"
# 	writeGif(filename, images, duration=0.5, subRectangles=False)

# if __name__ == '__main__':
# 	generate_gif()

def gen_video():
	images = []
	from glob import glob
	for img in glob('whiteboard_session/full/*.jpg'):
		images.append(Image.open(img))

	image1 = images[0]

	# Grab the stats from image1 to use for the resultant video
	height, width, layers =  numpy.array(image1).shape

	fourcc =cv2.cv.CV_FOURCC(*'XVID')

	# Create the OpenCV VideoWriter
	video = cv2.VideoWriter("meeting_review.avi", # Filename
	                        fourcc, # Negative 1 denotes manual codec selection. You can make this automatic by defining the "fourcc codec" with "cv2.VideoWriter_fourcc"
	                        30, # 10 frames per second is chosen as a demo, 30FPS and 60FPS is more typical for a YouTube video
	                        (width,height) # The width and height come from the stats of image1
	                        )

	# We'll have 30 frames be the animated transition from image1 to image2. At 10FPS, this is a whole 3 seconds
	img_count = 1
	while img_count != len(images):
		for i in xrange(0, 150):
			arr = cv2.cvtColor(numpy.array(images[img_count - 1]), cv2.COLOR_RGB2BGR)
			video.write(arr)
		for i in xrange(0,60):
		    images1And2 = Image.blend(images[img_count - 1], images[img_count], i/60.0)

		    # Conversion from PIL to OpenCV from: http://blog.extramaster.net/2015/07/python-converting-from-pil-to-opencv-2.html
		    video.write(cv2.cvtColor(numpy.array(images1And2), cv2.COLOR_RGB2BGR))

		img_count += 1

	for i in xrange(0, 150):
		#video.write(cv2.cvtColor(numpy.array(images[img_count - 1], cv2.COLOR_RGB2BGR)))
		arr = cv2.cvtColor(numpy.array(images[img_count - 1]), cv2.COLOR_RGB2BGR)
		video.write(arr)

	# Release the video for it to be committed to a file
	video.release()

if __name__ == '__main__':
	gen_video()