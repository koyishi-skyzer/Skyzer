import cv2
import numpy as np

capture=cv2.VideoCapture(0)
capture.set(10,160)
capture.set(3,1920)
capture.set(4,1080)
paper_w=297
paper_h=210
scale=3


#编写边缘检测函数
def getContours(img,ssize,msize,minArea=1000,filter=0,draw=False,showclose=False):
    '''该函数用于处理图像并提取轮廓'''
    img_1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_2 = cv2.GaussianBlur(img_1,(5,5),1)
    img_canny=cv2.Canny(img_2,ssize,msize)
    kernel = np.ones((5,5))
    img_3=cv2.dilate(img_canny,kernel,iterations=2)
    img_close=cv2.erode(img_3,kernel,iterations=2)
    if showclose == True:
        cv2.imshow('close',img_close)
    # cv2.imshow('canny',img_canny)

    #查找轮廓
    contours,hiearchy=cv2.findContours(img_close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    finalContours=[]

    for i in contours:
        area=cv2.contourArea(i)
        if area > minArea:
            para = cv2.arcLength(i,True)
            approx=cv2.approxPolyDP(i,0.02*para,True)   #通过该函数获得我们的矩形四角的点
            bBox=cv2.boundingRect(approx)
            # print(approx)
            if filter > 0:
                if len(approx) == filter:
                    finalContours.append([len(approx),area,approx,bBox,i])
            else:
                finalContours.append([len(approx),area,approx,bBox,i])

    finalContours=sorted(finalContours,key=lambda x:x[1],reverse=True)
    if draw:
        for i in finalContours:
            cv2.drawContours(img,i[4],-1,(0,0,255),3)
    
    return img , finalContours


#编写点的排序函数
def reorder(Points):
    '''该函数用于为A4纸四个点进行排序，以方便进行透视变换'''
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


#编写透视变换函数
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


#编写计算长度函数
def compute(point1,point2):
    ans=((point2[0]-point1[0])**2+(point2[1]-point1[1])**2)**0.5
    return ans


while True:
    ret,frame=capture.read()

    if ret == True:
        # cv2.imshow('1',frame)
        
        img,finalContours=getContours(frame,100,100,filter=4,draw=True)

        if len(finalContours)!=0:
            biggest=finalContours[0][2]
            # print(biggest)
            imgwarp=warp(img,biggest,paper_w,paper_h)
            # cv2.imshow('test',imgwarp) 
            img2,finalContours2=getContours(imgwarp,50,50,minArea=2000,filter=4,draw=True,showclose=True)   #对变换后的图片再次进行轮廓提取
            
            if len(finalContours) !=0:
                for i in finalContours2:
                    cv2.polylines(img2,[i[2]],True,(0,255,0),2)         #绘制线条使轮廓更加清晰
                    newpoint=reorder(i[2])
                    det_w=round(compute(newpoint[0][0],newpoint[1][0]))
                    det_h=round(compute(newpoint[0][0],newpoint[2][0]))
                    x,y,w,h=i[3]
                    # print('x:',x)
                    cv2.putText(img2, '{}mm'.format(det_w), (x+30,y-10) , cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255), 2)
                    cv2.putText(img2,'{}mm'.format(det_h),(x-70,y+h//2),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,255),2)
                    print('width:',det_w)
                    print('height:',det_h)
            cv2.imshow('contour',img2)
            
        img=cv2.resize(img,(0,0),None,0.5,0.5)
        cv2.imshow('original',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()