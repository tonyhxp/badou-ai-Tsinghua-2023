# coding: utf-8

import cv2


def coutImgAvg(gray):
    sum  = 0
    for i  in range (8):
        for j in range (8):
            sum += gray[i,j]
    return sum/64;


def creatHash(gray, avg):
    var_hash = ''
    for i in range(8):
        for j in range(8):
            if gray[i,j] > avg:
                var_hash =var_hash+'1'
            else:
                var_hash = var_hash+'0'
    return var_hash


def aHash(img):
    img = cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    avg_count = coutImgAvg(gray)
    hash_v = creatHash(gray,avg_count)
    return hash_v

def compareHash(hash1, hash2):
    if len(hash1) !=len(hash2):
        return -1

    n = 0
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n +=1
    return n


#均值哈希算法
img1 = cv2.imread('lenna.png')
img2 = cv2.imread('lenna_noise.png')
# hash值
hash1= aHash(img1)
hash2= aHash(img2)
print(hash1)
print(hash2)
# 计算相似度
p_like = compareHash(hash1,hash2)
print('均值hash算法相似度：',p_like)


#差值
def combos(gray):
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


def dHash(img):
    img=cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC) #w,h
    print(img.shape)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return combos(gray)

hash1= dHash(img1)
hash2= dHash(img2)
print(hash1)
print(hash2)
p_like=compareHash(hash1,hash2)
print('差值哈希算法相似度：',p_like)