# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. 加载数据（官方提供的手写数字图片）
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 2. 数据预处理（把图片变成模型能吃的格式）
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# 3. 创建模型（类似搭积木）
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 4. 训练模型（教AI认数字）
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=5, validation_split=0.2)

# 5. 测试准确率
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'测试准确率: {test_acc*100:.2f}%')  # 正常应显示98%左右

# 6. 保存模型（方便以后使用）
model.save('my_mnist_model.h5')

# 7. 画训练曲线（会弹出图像窗口）
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.legend()
plt.show()