# to be done
import os
import cv2
import glob

images_path = '/Users/smy1999/Desktop/leftimg8bit_video/leftImg8bit/demoVideo/stuttgart_02/'
num = 50

images = glob.glob(images_path + '*.png')
images.sort()

video = cv2.VideoWriter('/Users/smy1999/Desktop/test.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 10, (1280, 720))  #定义保存视频目录名称及压缩格式，fps=10,像素为1280*720
for i, img in enumerate(images):
    img = cv2.imread(img)  # 读取图片
    img = cv2.resize(img, (1280, 720))  # 将图片转换为1280*720
    video.write(img)  # 写入视频
    if i > num:
        break

video.release()
