import numpy as np
import cv2  


original = cv2.imread('image.jpg')
changed = cv2.imread('image_removed.jpg')

img_a = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
img_a = cv2.GaussianBlur(img_a, (11, 11), 0)

img_b = cv2.cvtColor(changed, cv2.COLOR_BGR2GRAY)
img_b = cv2.GaussianBlur(img_b, (11, 11), 0)

img_diff = cv2.absdiff(img_a, img_b)

thresh = cv2.threshold(img_diff, 15, 255, cv2.THRESH_BINARY)[1]

# dilate the thresholded image to fill in holes, then find contours
# on thresholded image
thresh = cv2.dilate(thresh, None, iterations=2)
(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)

contours_final = []

# loop over the contours
for c in cnts:

	# compute the bounding box for the contour, draw it on the frame,
	# and update the text
	(x, y, w, h) = cv2.boundingRect(c)
	hey = cv2.rectangle(img_diff, (x, y), (x + w, y + h), (255, 0, 0), 2)
	contours_final.append(hey)

cv2.drawContours( img_a, contours_final, -1, (255, 0, 0), 3 )
cv2.imshow('im',original)
cv2.imshow('im removed', changed)

ch = 0xFF & cv2.waitKey()
cv2.destroyAllWindows()

cv2.imshow('imdf',img_diff)

ch = 0xFF & cv2.waitKey()
cv2.destroyAllWindows()