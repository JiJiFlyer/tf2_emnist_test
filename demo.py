from alfred.dl.tf.common import mute_tf
mute_tf()
import os
import cv2
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from alfred.utils.log import logger as logging
#import tensorflow_datasets as tfds
from model import LeNet_inference, AlexNet_inference, ResNet_inference
import glob


target_size = 28
num_classes = 62
# use_keras_fit = False
use_keras_fit = True
ckpt_path = './checkpoints/ResNet/ResEpoch-{epoch}.ckpt'

def load_characters():
    a = open('characters.txt', 'r',encoding='UTF-8').readlines()
    return [i.strip() for i in a]
characters = load_characters() # 载入标签向量矩阵


def get_model():
    # init model
    model = ResNet_inference((28, 28, 1), num_classes, 0.5)
    logging.info('model loaded.')

    latest_ckpt = tf.train.latest_checkpoint(os.path.dirname(ckpt_path))
    if latest_ckpt:
        start_epoch = int(latest_ckpt.split('-')[1].split('.')[0])
        model.load_weights(latest_ckpt)
        logging.info('model resumed from: {} at epoch: {}'.format(latest_ckpt, start_epoch))
        return model
    else:
        logging.error('can not found any checkpoints matched: {}'.format(ckpt_path))


#--------------------------#
# 对读取的图片预处理
#--------------------------#
def pre_pic(picName):
    # reIm = picName.resize((target_size,target_size), Image.ANTIALIAS)
    im_arr = np.array(picName)
    # 对图片做二值化处理（滤掉噪声，threshold调节阈值）
    threshold = 25
    # 模型的要求是黑底白字，但输入的图是白底黑字，所以需要对每个像素点的值改为255减去原值以得到互补的反色。
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if (im_arr[i][j] < threshold):
                im_arr[i][j] = 0
            else: im_arr[i][j] = 255
    # 把图片形状拉成1行784列，并把值变为浮点型（因为要求像素点是0-1 之间的浮点数）
    nm_arr = im_arr.reshape([1, 784])
    nm_arr = nm_arr.astype(np.float32)
    # 接着让现有的RGB图从0-255之间的数变为0-1之间的浮点数
    img_ready = np.multiply(nm_arr, 1.0/255.0)
    img_ready = tf.reshape(img_ready,[28, 28])
    # plt.imshow(img_ready)
    # plt.show()
    return img_ready


def predict(model, img_f):
    img_ready = cv2.imread(img_f)
    img = tf.expand_dims(img_ready[:, :, 0], axis=-1)
    img = tf.image.resize(img, (target_size, target_size))
    
    img = pre_pic(img)

    img = tf.expand_dims(img, axis=-1)
    img = tf.transpose(img) # 训练集集预处理没做好，这里需要旋转镜像
    # img = tf.image.rot90(img, k=1) 
    img = tf.expand_dims(img, axis=-1)
    print(img.shape)
    result = model.predict(img)
    print('predict: {}'.format(characters[np.argmax(result[0])]))

    name = 'assets/pred_{}.png'.format(characters[np.argmax(result[0])])
    #cv2.imwrite(name, ori_img)  #路径中文会乱码
    cv2.imencode('.jpg', img_ready)[1].tofile(name) #正确的解决办法

if __name__ == '__main__':
    img_files = glob.glob('assets/*.png')
    model = get_model()
    for img_f in img_files:
        a = cv2.imread(img_f)
        cv2.imshow('monitor', a)
        cv2.moveWindow("monitor",960,540)
        predict(model, img_f)
        cv2.waitKey(0)


