#!/usr/bin/env python
# encoding=gbk


import cv2
import numpy as np

img = cv2.imread('../image/mm.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("canny_simple",cv2.Canny(gray,200,300))
cv2.waitKey()
cv2.destroyAllWindows()