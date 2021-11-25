import cv2
import math

#导入图片
# path="./angleTest.jpg"
path="C:\\Users\\admin\\Desktop\\angleTest.jpg"
img=cv2.imread(path)
pointsList=[]

#创建鼠标回调函数
def mouseEvent(event,x,y,flags,params):
    if event ==cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y),5,(255,0,0),-1)
        print(x,y)
        pointsList.append([x,y])
        print(pointsList)
        pass

#创建梯度计算函数
def gradient(p1,p2):
    if p2[0]-p1[0]!=0:
        return (p2[1]-p1[1])/(p2[0]-p1[0])
    if p2[0]-p1[0]==0:
        return 0

#创建角度计算函数
def getAngle(pointlist):
    if len(pointsList)%3 == 0 and len(pointsList)!=0:
        p1,p2,p3=pointlist[-3:]
        m1=gradient(p1,p2)
        m2=gradient(p1,p3)
        rad=math.atan((m2-m1)/(1+m2*m1))
        ang=round(math.degrees(rad))
        print(ang)
        # if ang <= 0:
        #     print(180+ang)

while True:
    cv2.imshow('image',img)
    cv2.setMouseCallback('image',mouseEvent)
    getAngle(pointsList)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pointsList=[]
        img=cv2.imread(path)
