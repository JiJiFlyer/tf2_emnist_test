import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import cv2
import numpy
from PIL import Image

EPOCH = 6
LOSS_RATE = 0.01
OUTPUT_CLASS = 62
SHUFFLE_SIZE = 500
BATCH_SIZE = 128
VALI_STEPS = 980
STEPS_PER_EPOCH = 1000

ckpt_path = './checkpoints/pro1-{epoch}.ckpt'
def load_characters():
    a = open('characters.txt', 'r',encoding='UTF-8').readlines()
    return [i.strip() for i in a]
characters = load_characters() # 载入标签向量矩阵

#选择指定显卡及自动调用显存
# physical_devices = tf.config.experimental.list_physical_devices('GPU')#列出所有可见显卡
# print("All the available GPUs:\n",physical_devices)
# if physical_devices:
#     gpu=physical_devices[0]#显示第一块显卡
#     tf.config.experimental.set_memory_growth(gpu, True)#根据需要自动增长显存
#     tf.config.experimental.set_visible_devices(gpu, 'GPU')#只选择第一块

# Build the `tf.data.Dataset` pipeline.
(ds_train, ds_test), ds_info = tfds.load(
    'emnist',
    split=['train', 'test'],
    data_dir='./tensorflow_datasets',
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  tf.transpose(image)   # emnist数据集天生旋转，需要摆正
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
image = list(ds_train.take(5))[2] # 取前5张的第3张
a = image[0][:,:,0] # 0为image
print(image[0])
print(characters[image[1].numpy()]) # 1为label
a = tf.transpose(a)
#plt.imshow(a, cmap='gray') 
plt.imshow(a, cmap='gray_r')  #此为输入模型的图片
plt.show()

a.save('assets/0.png')

