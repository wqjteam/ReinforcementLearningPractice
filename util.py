import cv2
import numpy as np
import matplotlib.pyplot as plt # plt 用于显示图片


def preprocess(observation):
    """
    image preprocess
    :param observation:
    :return:
    """
    observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    #调用函数将彩色图像转灰度图像(resize是进行给定尺度的缩放，）
    # plt.figure(2)
    # plt.imshow(observation)
    observation = observation[26:110,:]         #提取部分将通道尺寸变成方形
    # plt.figure(3)
    # plt.imshow(observation)
    ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    #cv2.threshold(原图像,阈值，固定值，cv2.THRESH_BINARY代表原图像素只要大于阈值，则都取固定值）
    x = np.reshape(observation,(84,84,1))
    # plt.figure(4)
    # plt.imshow(x)
    return x.transpose((2, 0, 1))    #transpose 是将数据结构进行转置https://blog.csdn.net/u012762410/article/details/78912667
