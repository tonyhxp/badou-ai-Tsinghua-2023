import cv2
import numpy as np

img = cv2.imread('../image/mm.png')
res = img.copy()

src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

matric = cv2.getPerspectiveTransform(src,dst)
print(matric)

lst = cv2.warpPerspective(res, matric, (0,0))
cv2.imshow("src",img)
cv2.imshow("dst",lst)
cv2.waitKey()
cv2.destroyAllWindows()
