# coding: utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../image/lenna.png', 0)

rows, clos = img.shape[:]

data = img.reshape((rows * clos, 1))
data = np.float32(data)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            10,
            1.0)

flag = cv2.KMEANS_PP_CENTERS

comactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flag)

dst = labels.reshape((img.shape[0], img.shape[1]))


#用来正常显示中文标签 中文黑体字
plt.rcParams['font.sans-serif']=['SimHei']

titles = [u'原始图像', u'聚类图像']
images = [img, dst]
for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray'),
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
