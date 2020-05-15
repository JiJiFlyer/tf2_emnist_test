import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

def LeNet_inference(in_shape, outputclass, dropout):
    model = models.Sequential(name='LeNet')
    # 第1层卷积，卷积核大小为3*3，32个，28*28为待训练图片的大小
    model.add(layers.Conv2D(
        32, (3, 3), padding='same', activation='relu', input_shape=(in_shape[0],in_shape[1],in_shape[2])))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    # 第2层卷积，卷积核大小为3*3，64个
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    # 第3层卷积，卷积核大小为3*3，64个
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

    model.add(layers.Flatten())
    model.add(layers.Dense(80, activation='relu'))
    model.add(layers.Dense(outputclass, activation='softmax'))

    model.summary()
    #25s 13ms/step - loss: 0.3786 - accuracy: 0.8615 - val_loss: 0.4692 - val_accuracy: 0.8417
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(dropout),
        metrics=['accuracy'],
    )

    return model



def AlexNet_inference(in_shape, outputclass, dropout):
    model = models.Sequential(name='AlexNet')  
 
    # model.add(layers.Conv2D(96,(11,11),strides=(4,4),input_shape=(in_shape[1],in_shape[2],in_shape[3]),
                # padding='same',activation='relu',kernel_initializer='uniform')) 
                
    model.add(layers.Conv2D(96,(11,11),strides=(2,2),input_shape=(in_shape[0],in_shape[1],in_shape[2]),
                padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(layers.MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
    model.add(layers.Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(layers.MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
    model.add(layers.Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(layers.Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(layers.Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
    model.add(layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))  
    model.add(layers.Flatten())  
    model.add(layers.Dense(2048,activation='relu'))  
    model.add(layers.Dropout(dropout))  
    model.add(layers.Dense(2048,activation='relu'))  
    model.add(layers.Dropout(dropout))  
    model.add(layers.Dense(outputclass,activation='softmax'))  
    model.compile(optimizer=keras.optimizers.Adam(),
                loss='sparse_categorical_crossentropy', #不能直接用函数，否则在与测试加载模型不成功！
                metrics=['accuracy'])
    model.summary()  
    # 75s 38ms/step - loss: 0.6148 - accuracy: 0.8204 - val_loss: 0.5183 - val_accuracy: 0.8379
    return model

def VGG16_inference(in_shape, outputclass, dropout):
    network_VGG16 = Sequential([
        # 第一层
        layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu'),
        # 第二层
        layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=2, strides=2),
        # 第三层
        layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu'),
        # 第四层
        layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=2, strides=2),
        # 第五层
        layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu'),
        # 第六层
        layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu'),
        # 第七层（新增卷积层1*1*256）
        layers.Conv2D(256, kernel_size=1, strides=1, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=2, strides=2),
        # 第八层
        layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),
        # 第九层
        layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),
        # 第十层（新增卷积层1*1*512）
        layers.Conv2D(512, kernel_size=1, strides=1, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=2, strides=2),
        # 第十一层
        layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),
        # 第十二层
        layers.Conv2D(512, kernel_size=3, strides=1, padding='same', activation='relu'),
        # 第十三层（新增卷积层1*1*512）
        layers.Conv2D(512, kernel_size=1, strides=1, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=2, strides=2),
        layers.Flatten(),  # 拉直 7*7*512
        # 第十四层
        layers.Dense(1024, activation='relu'),
        layers.Dropout(rate=dropout),
        # 第十五层
        layers.Dense(128, activation='relu'),
        layers.Dropout(rate=dropout),
        # 第十六层
        layers.Dense(5, activation='softmax')
    ])
    network_VGG16.compile(optimizer=keras.optimizers.Adam(),
            loss='sparse_categorical_crossentropy', #不能直接用函数，否则在与测试加载模型不成功！
            metrics=['accuracy'])
    network_VGG16.summary()  # 打印各层参数表
    return network_VGG16

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