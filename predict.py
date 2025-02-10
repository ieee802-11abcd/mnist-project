import cv2
import numpy as np
import tensorflow as tf

# 加载训练好的模型
model = tf.keras.models.load_model('my_mnist_model.h5')

# 读取图片并预处理
img = cv2.imread('test_num.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28,28))  # 确保尺寸正确
img = img.reshape(1,28,28,1).astype('float32') / 255

# 预测并显示结果
prediction = model.predict(img)
print(f'预测结果: {np.argmax(prediction)}')