from alfred.dl.tf.common import mute_tf
mute_tf()
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
import datetime as dt
import os
from alfred.utils.log import logger as logging
import model

use_keras_callbacks = True
EPOCH = 20
DROPOUT = 0.5
OUTPUT_CLASS = 62
SHUFFLE_SIZE = 500
BATCH_SIZE = 128
VALI_STEPS = 1950
STEPS_PER_EPOCH = 2000
SAVE_PERIOD = 10
ckpt_path = './checkpoints/ResNet/ResEpoch-{epoch}.ckpt'
log_dir = os.path.join(
    "tblogs",
    "resfit",
    dt.datetime.now().strftime("%Y%m%d-%H%M%S"),
) # a Windows-specific bug in TensorFlow, no hard-coding path
MODEL_DIR ='./checkpoints/' # use_keras_callbacks = False时的路径

# 选择指定显卡及自动调用显存
physical_devices = tf.config.experimental.list_physical_devices('GPU')#列出所有可见显卡
print("All the available GPUs:\n",physical_devices)
if physical_devices:
    gpu=physical_devices[0]#显示第一块显卡
    tf.config.experimental.set_memory_growth(gpu, True)#根据需要自动增长显存
    tf.config.experimental.set_visible_devices(gpu, 'GPU')#只选择第一块

# Build the `tf.data.Dataset` pipeline.
(ds_train, ds_test), ds_info = tfds.load(
    'emnist',
    split=['train', 'test'],
    data_dir='./tensorflow_datasets',
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# 数据集预处理
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  tf.transpose(image)   # emnist数据集天生旋转，需要摆正
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # num_parallel_calls设置多线程处理
inputshape = list(ds_train.take(1))[0][0].shape
print(inputshape)   # 打印输入图像大小


# 载入模型
model = model.ResNet_inference(inputshape, OUTPUT_CLASS, DROPOUT)

logging.info('model loaded.')

start_epoch = 0
latest_ckpt = tf.train.latest_checkpoint(os.path.dirname(ckpt_path))
if latest_ckpt:
    start_epoch = int(latest_ckpt.split('-')[1].split('.')[0])
    model.load_weights(latest_ckpt)
    logging.info('model resumed from: {}, start at epoch: {}'.format(latest_ckpt, start_epoch))
else:
    logging.info('passing resume since weights not there. training from scratch')

# 模型训练
if use_keras_callbacks:

    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE)
    ds_train = ds_train.repeat()
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.cache()
    ds_test = ds_test.shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE)
    ds_test = ds_test.repeat()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                            save_weights_only=True,
                                            verbose=1,
                                            period=SAVE_PERIOD,
                                            #save_freq=50000
        ),
        tf.keras.callbacks.TensorBoard(log_dir, write_graph=True, write_images=True)
    ]
    # try:
    model.fit(
        ds_train,
        epochs=EPOCH,
        validation_data=ds_test,
        validation_steps=VALI_STEPS,
        steps_per_epoch=STEPS_PER_EPOCH,
        callbacks=callbacks
    )
    # except KeyboardInterrupt:
    model.save(os.path.join(os.path.dirname(ckpt_path), 'ResNet_proj1.h5'))

else: # 不用callbacks的训练控制，用作学习笔记，慎用
    totall_epochs = 0
    epochs = 10
    while(True):
 
        history = model.fit(ds_train, batch_size=BATCH_SIZE, epochs=epochs, validation_split=0.1)
 
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.legend(['training', 'valivation'], loc='upper left')
        plt.show()
 
        res = model.evaluate(ds_test)
        print(res)
        
        totall_epochs += epochs
        model_save_dir = MODEL_DIR+'AlexNet_model_'+str(totall_epochs)+'.h5'
        AlexNet_model.save( model_save_dir )
 
        keyVal = input('please enter your command!(0:quite, 1>:continue these epochs!)')
        keyVal = int(keyVal)
        if 0==keyVal:
            break
        elif 0<=keyVal and 50>=keyVal:
            epochs = keyVal