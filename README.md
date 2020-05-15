# TensorFlow 2 EMNIST数据集上的ResNet字母数字识别模型
本项目在window10 + tf2.1 + python3.7环境下运行良好。

## 数据准备

数据全部来自于TF官网收录的EMNIST数据集：
- https://tensorflow.google.cn/datasets/catalog/emnist?hl=en
- emnist/byclass (default config)：本项目使用默认的62分类数据集，分别为10个数字，26个大小写字母（10+26+26），训练集697932个，测试集116323个

读取（路径没有数据则自动下载）数据集：
```python
(ds_train, ds_test), ds_info = tfds.load(
    'emnist',
    split=['train', 'test'],
    data_dir='./tensorflow_datasets',
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
) 
```
这部分在mytrain.py，参考自官方文档https://tensorflow.google.cn/datasets/keras_example


## 模型构建

使用keras构建了4个CNN模型分别做测试，最后选用了效果最好的ResNet，模型构建写在model.py。


```python
#--------------------------#
# RESNET
#--------------------------#
def res_net_block(input_data, filters, conv_size):
  # CNN层
  x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
  x = layers.BatchNormalization()(x)
  x = layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
  # 第二层没有激活函数
  x = layers.BatchNormalization()(x)
  # 两个张量相加
  x = layers.Add()([x, input_data])
  # 对相加的结果使用ReLU激活
  x = layers.Activation('relu')(x)
  # 返回结果
  return x

def ResNet_inference(input_shape, n_classes, dropout):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(3)(x)
    num_res_net_blocks = 10
    for i in range(num_res_net_blocks):
        x = res_net_block(x, 64, 3)
    # 添加一个CNN层
    x = layers.Conv2D(64, 3, activation='relu')(x)
    # 全局平均池化GAP层
    x = layers.GlobalAveragePooling2D()(x)
    # 几个密集分类层
    x = layers.Dense(256, activation='relu')(x)
    # 退出层
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    res_net_model = keras.Model(inputs, outputs)
    res_net_model.compile(optimizer=keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    res_net_model.summary()
    #  105s 53ms/step - loss: 0.2584 - accuracy: 0.8978 - val_loss: 0.3838 - val_accuracy: 0.8743
    #  2000*20steps开始过拟合
    return res_net_model
```



## 训练模型

- 使用keras.models.fit来训练模型
```python
callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path,
                                            save_weights_only=True,
                                            verbose=1,
                                            period=SAVE_PERIOD),
        tf.keras.callbacks.TensorBoard(log_dir, write_graph=True, write_images=True)
    ]
```

运行mytrain.py开始训练，会自动下载数据集（根目录生成tensorflow_datasets文件夹）

- 使用tf.keras.callbacks.ModelCheckpoint实现训练中自动保存。
- 支持断点续训，退出时直接保存模型，再次训练会读取最近保存的weight。
- TensorBoard用于实时监测训练情况，根目录下命令行输入tensorboard --logdir=tblogs,按提示打开浏览器查看

数据集较大，建议使用GPU训练。


## 模型测试
将待识别图片放入assets（支持白底彩字），运行demo.py即可，默认识别PNG，想识别jpg修改一下demo.py就可以
```python
if __name__ == '__main__':
    img_files = glob.glob('assets/*.png')
    model = get_model()
    for img_f in img_files:
        a = cv2.imread(img_f)
        cv2.imshow('monitor', a)
        cv2.moveWindow("monitor",960,540)
        predict(model, img_f)
        cv2.waitKey(0)
```
运行时会逐一弹窗，并打印识别的字，文件夹中会生成由字母本身命名的图片。
![Image text](https://github.com/JiJiFlyer/tf2_emnist_test/blob/master/imgs/demo.png)
