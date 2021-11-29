import cv2
import numpy as np
import datetime
import math


def array_to_tuple(arr):
    return tuple(arr.reshape(1, -1)[0])


# 颜色字典
colors = {'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255), 'yellow': (0, 255, 255),
          'magenta': (255, 0, 255), 'cyan': (255, 255, 0), 'white': (255, 255, 255), 'black': (0, 0, 0),
          'gray': (125, 125, 125), 'rand': np.random.randint(0, high=256, size=(3,)).tolist(),
          'dark_gray': (50, 50, 50), 'light_gray': (220, 220, 220)}

# 创建画布，属性为: 640 x 640 pixels, 3 channels, uint8 (8-bit unsigned integers)  
# 使用np.zeros()使得背景为黑色
image = np.zeros((680, 680, 3), dtype="uint8")

#(改变背景颜色（图案）)
image[:] = cv2.imread('C:\\Users\\admin\\Desktop\\04.jpg',1)   #设置背景为图片则图片大小需要和我们所建立的画布大小一致
# image[:] = colors['light_gray']                              #改变背景颜色

hours_orig = np.array(
    [(620, 320), (580, 470), (470, 580), (320, 620), (170, 580), (60, 470), (20, 320), (60, 170), (169, 61), (319, 20),
     (469, 60), (579, 169)])

hours_dest = np.array(
    [(600, 320), (563, 460), (460, 562), (320, 600), (180, 563), (78, 460), (40, 320), (77, 180), (179, 78), (319, 40),
     (459, 77), (562, 179)])


# We draw the hour markings:
for i in range(0, 12):
    cv2.line(image, array_to_tuple(hours_orig[i]), array_to_tuple(hours_dest[i]), colors['red'], 3)
#cv2.line()函数,函数原型：img=line(img,pt1,pt2,color,thickness)
               #函数在连接pt1和pt2的img图像上画一条线 
               
#画一个圆来模拟时钟的形状
cv2.circle(image, (320, 320), 310, colors['dark_gray'], 8)

#创建一个写有"Mastering OpenCV 4 with Python"的矩阵文本:    
cv2.rectangle(image, (150, 175), (490, 270), colors['dark_gray'], -1)
cv2.putText(image, "Mastering OpenCV 4", (150, 200), 1, 2, colors['light_gray'], 1, cv2.LINE_AA)
cv2.putText(image, "with Python", (210, 250), 1, 2, colors['light_gray'], 1, cv2.LINE_AA)

# 用"静态"信息来复制图像信息
image_original = image.copy()

# 绘制动态信息:
while True:
    # 获取当前日期:
    date_time_now = datetime.datetime.now()
    # 从日期获取当前时间:
    time_now = date_time_now.time()
    # 从时间获取当前时分秒:
    hour = math.fmod(time_now.hour, 12)
    minute = time_now.minute
    second = time_now.second

    print("hour:'{}' minute:'{}' second: '{}'".format(hour, minute, second))

    # 获取时分秒角度:
    second_angle = math.fmod(second * 6 + 270, 360)
    minute_angle = math.fmod(minute * 6 + 270, 360)
    hour_angle = math.fmod((hour * 30) + (minute / 2) + 270, 360)

    print("hour_angle:'{}' minute_angle:'{}' second_angle: '{}'".format(hour_angle, minute_angle, second_angle))

    # 画出时分秒对应的线
    second_x = round(320 + 310 * math.cos(second_angle * 3.14 / 180))
    second_y = round(320 + 310 * math.sin(second_angle * 3.14 / 180))
    cv2.line(image, (320, 320), (second_x, second_y), colors['blue'], 2)

    minute_x = round(320 + 260 * math.cos(minute_angle * 3.14 / 180))
    minute_y = round(320 + 260 * math.sin(minute_angle * 3.14 / 180))
    cv2.line(image, (320, 320), (minute_x, minute_y), colors['blue'], 8)

    hour_x = round(320 + 220 * math.cos(hour_angle * 3.14 / 180))
    hour_y = round(320 + 220 * math.sin(hour_angle * 3.14 / 180))
    cv2.line(image, (320, 320), (hour_x, hour_y), colors['blue'], 10)

    # 最后画一个小圆，位于三个指针的交界处:
    cv2.circle(image, (320, 320), 10, colors['dark_gray'], -1)
    cv2.imshow("clock", image)
    image = image_original.copy()

    if cv2.waitKey(500) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()