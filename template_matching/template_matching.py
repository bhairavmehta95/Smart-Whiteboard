import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('big.jpg',0)
img2 = img.copy()
templates = []
template = cv2.imread('small.jpg',0)
templates.append(template)
template = cv2.imread('small2.jpg',0)
templates.append(template)
template = cv2.imread('small3.jpg',0)
templates.append(template)
w, h = template.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for picture in templates:
  cv2.imshow('Searching for: ', picture)
  ch = 0xFF & cv2.waitKey()
  cv2.destroyAllWindows()

  for meth in methods:
    img = img2.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img,picture,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
      top_left = min_loc
    else:
      top_left = max_loc
      bottom_right = (top_left[0] + w, top_left[1] + h)
     
    cv2.rectangle(img,top_left, bottom_right, 0, 2)
    print top_left, bottom_right
    cv2.imshow(str(meth),img)

    ch = 0xFF & cv2.waitKey()
    cv2.destroyAllWindows()
