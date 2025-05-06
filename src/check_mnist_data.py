import mnist_loader

# データを読み込む
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# 訓練データの最初の1件を確認
first_input, first_label = training_data[0]

print("=== 訓練データの1件目 ===")
print("画像データ shape:", first_input.shape)
print("画像データ（最初の10個）:", first_input[:10].flatten())
print("ラベル（one-hot ベクトル）:\n", first_label)

# テストデータの最初の1件を確認
first_test_input, first_test_label = test_data[0]

print("\n=== テストデータの1件目 ===")
print("画像データ shape:", first_test_input.shape)
print("画像データ（最初の10個）:", first_test_input[:10].flatten())
print("ラベル（数字）:", first_test_label)
