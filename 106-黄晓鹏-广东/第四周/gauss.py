import cv2
from skimage import util

img = cv2.imread("../image/mm.png")
noise_gs_img=util.random_noise(img,mode='poisson')

cv2.imshow("source", img)
cv2.imshow("lenna",noise_gs_img)
cv2.waitKey()
cv2.destroyAllWindows()
