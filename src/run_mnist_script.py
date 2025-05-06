import mnist_loader
import network

def main():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network([784, 100, 10])

    # 初期化直後の重み・バイアスを表示
    print("=== 初期バイアス ===")
    for i, b in enumerate(net.biases):
        print(f"Layer {i+2} bias shape: {b.shape}")
        print(b)
    print("\n=== 初期重み ===")
    for i, w in enumerate(net.weights):
        print(f"Layer {i+1}->{i+2} weight shape: {w.shape}")
        print(w)

    # 学習前の正解率を確認
    initial_accuracy = net.evaluate(test_data)
    print(f"=== 学習前の正解率 ===\n{initial_accuracy} / {len(test_data)}")

    # 学習を実行
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

    # 学習後の重み・バイアスを表示
    print("\n=== 学習後のバイアス ===")
    for i, b in enumerate(net.biases):
        print(f"Layer {i+2} bias shape: {b.shape}")
        print(b)
    print("\n=== 学習後の重み ===")
    for i, w in enumerate(net.weights):
        print(f"Layer {i+1}->{i+2} weight shape: {w.shape}")
        print(w)

if __name__ == "__main__":
    main()
