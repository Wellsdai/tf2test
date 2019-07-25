from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

print(tf.__version__)
print(np.__version__ )
print(tf.keras.__version__)

#第一步。载入 MNIST 数据集，并将整型转换为浮点型，除以 255 是为了归一化。
#也可以使用 TensorFlow 新出的 tensorflow-datasets 载入数据集，见https://tensorflow.google.cn/datasets/overview
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train,x_test = x_train/255.0,x_test/255.0

#第二步。使用 tf.keras.Sequential 建立模型
"""
1.Sequential 用于建立顺序模型
2.Flatten 层用于展开张量，input_shape 定义输入形状为 28x28 的图像，展开后为 28*28 的张量。
3.Dense 层为全连接层，输出有 128 个神经元，激活函数使用 relu。
4.Dropout 层使用 0.2 的失活率。解决过拟合问题
5.再接一个全连接层，激活函数使用 softmax，得到对各个类别预测的概率。
"""
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

#第三步。用keras的complie来配置学习过程，编译模型
#优化器选择 Adam 优化器。
#损失函数使用 sparse_categorical_crossentropy，
"""
还有一个损失函数是 categorical_crossentropy，
两者的区别在于输入的真实标签的形式，sparse_categorical 输入的是整形的标签，
例如 [1, 2, 3, 4]，categorical 输入的是 one-hot 编码的标签。
"""
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#第四步。训练评估模型。 对训练数据遍历一次为一个 epoch，这里遍历 5 次。
model.fit(x_train,y_train,epochs=5)

#第五步。 evaluate 用于评估模型，返回的数值分别是损失和指标。
model.evaluate(x_test,y_test)

#台式机修改