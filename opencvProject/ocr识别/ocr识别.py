import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image


#读取图像
image=cv2.imread("word.jpg",1)
image_copy=image.copy()
print(image.shape)
# cv2.imshow('origin',image)
ratio=image.shape[0]/800

def resize(img,height=None,width=None,inter=cv2.INTER_AREA):
    '''
    图片尺寸变换函数
    '''
    i=None
    (h,w)=img.shape[0:2]
    if width==None and height==None:
        return img
    elif width==None and height!=None:
        rat=height/h
        dim=(int(w*rat),height)
    elif width!=None and height==None:
        rat=width/w
        dim=(width,int(rat*h))
    else:
        dim=(width,height)

    resized=cv2.resize(img,dim,interpolation=inter)
    return resized

def getContours(img,ssize,msize):
    '''
    轮廓提取函数
    '''
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_gauss=cv2.GaussianBlur(img_gray,(5,5),1)
    img_canny=cv2.Canny(img_gauss,ssize,msize)
    cv2.imshow('canny',img_canny)

    contours,hiearchy=cv2.findContours(img_canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    contours=sorted(contours,key=cv2.contourArea,reverse=True)[:5]

    for i in contours:
        para=cv2.arcLength(i,True)
        approx=cv2.approxPolyDP(i,0.02*para,True)
        if len(approx)==4:
            fin_cnt=approx
    print(fin_cnt)
    return fin_cnt

#编写点的排序函数
def reorder(Points):
    '''该函数用于为轮廓四个点进行排序，以方便进行透视变换'''
    # print(Points.shape)
    newPoints=np.zeros_like(Points)
    Points=Points.reshape((4,2))
    add=Points.sum(1)
    newPoints[0]=Points[np.argmin(add)]
    newPoints[3]=Points[np.argmax(add)]
    diff=np.diff(Points,axis=1)             #求矩阵的离散差值，a[n]-a[n-1]
    newPoints[1]=Points[np.argmin(diff)]
    newPoints[2]=Points[np.argmax(diff)]
    return newPoints

def warp(img,points,w,h):
    '''透视变换函数'''
    # print('origin:',points)
    # print('new:',reorder(points))
    points=reorder(points)
    p_1=np.float32(points)
    p_2=np.float32([[0,0],[w,0],[0,h],[w,h]])
    gpt=cv2.getPerspectiveTransform(p_1,p_2)
    warpimg=cv2.warpPerspective(img,gpt,(w,h))
    return warpimg

tranImage=resize(image,height=800)
# print(tranImage.shape)

cnts=getContours(tranImage,50,250)
cv2.drawContours(tranImage,[cnts],-1,(0,0,255),2)
newpoints=reorder(cnts)
# imgwarp=warp(tranImage,cnts,600,800)
imgwarp=warp(image_copy,cnts*ratio,650,900)
imgwarp=cv2.cvtColor(imgwarp,cv2.COLOR_BGR2GRAY)
ret=cv2.threshold(imgwarp,105,255,cv2.THRESH_BINARY)[1]
# print(newpoints)
cv2.imshow('1',tranImage)
cv2.imshow('2',ret)
# cv2.imwrite('train.jpg',ret)

# #-------------------------------------------------------
# hist=cv2.calcHist([imgwarp],[0],None,[256],[0,256])
# plt.figure(figsize=(4,3))
# plt.suptitle("Grayscale",fontsize=5)
# plt.xlabel("bins")
# plt.ylabel("number of pixels")
# plt.xlim([0,256])
# plt.plot(hist,color='red')
# plt.show()
# #-------------------------------------------------------
cv2.waitKey(0)
text=pytesseract.image_to_string(Image.fromarray(ret),lang="chi_sim")
print(text)


