import matplotlib.pyplot as plt
import numpy as np
import mnist_loader

# データ読み込み
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# 訓練データの1件目
first_image = training_data[0][0]  # (784, 1)
first_label = np.argmax(training_data[0][1])  # one-hot → 数字に変換

# (784, 1) → (28, 28) に変換
image_2d = first_image.reshape(28, 28)

# 画像表示
plt.imshow(image_2d, cmap='gray')
plt.title(f"Label: {first_label}")
plt.axis('off')
plt.show()
