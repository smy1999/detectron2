# to be done
import os
import cv2
import glob

images_path = '/Users/smy1999/Desktop/demoVideo/stuttgart_00/'


images = glob.glob(images_path + '*.png')
images.sort()
num = len(images)
print(num)
video = cv2.VideoWriter(f'/Users/smy1999/Desktop/test{num}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (1280, 720))  #定义保存视频目录名称及压缩格式，fps=10,像素为1280*720
for i, img in enumerate(images):
    if i >= num:
        break
    img = cv2.imread(img)  # 读取图片
    img = cv2.resize(img, (1280, 720))  # 将图片转换为1280*720
    video.write(img)  # 写入视频


video.release()
