import random
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

class Network():
    def __init__(self, sizes):
        # 网络层熟
        self.num_layers = len(sizes)
        # 网络每层神经元个数
        self.sizes = sizes
        # 初始化每层的偏置和权重
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)

        return a

    # 随机梯度下降
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data: n_test = len(test_data)
        # 训练数据总个数
        n = len(training_data)

        # 开始训练 循环每一个epochs
        for  j in range(epochs):
            # 洗牌打乱数据
            random.shuffle(training_data)

            # mini_batch
            mini_batchs = [training_data[k: k + mini_batch_size]
                           for k in range(0, n, mini_batch_size)]

            # 训练mini_batch
            for mini_batch in mini_batchs:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test
                ))
            print("Epoch {0} complte".format(j))

    # 更新mini_batch
    def update_mini_batch(self, mini_batch, eta):
        # 保存每层的偏导
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 训练每一个mini_batch
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.update(x, y)

    # 前向传播
    def update(self, x, y):
        # 保存每层偏导
