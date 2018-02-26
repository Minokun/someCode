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
    def SGD(self, training_data, train_labels, epochs, mini_batch_size, eta, test_data=None, test_labels=None):
        if test_data: n_test = len(test_data)
        # 训练数据总个数
        n = len(training_data)

        # 开始训练 循环每一个epochs
        for  j in range(epochs):
            # 洗牌打乱数据 获取1 - len(training_data)随机十分之一的数字

            index_num = random.sample(range(n), int(n / 10))
            index_num.sort()

            # mini_batch
            mini_batchs = [training_data[k]
                           for k in index_num]

            mini_batchs_labels = [train_labels[k]
                           for k in index_num]

            # 训练mini_batch
            for mini_batch, mini_batchs_label in zip(mini_batchs, mini_batchs_labels):
                self.update_mini_batch(mini_batch, mini_batchs_label, eta)

            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data, test_labels), n_test
                ))
            print("Epoch {0} complte".format(j))

    # 更新mini_batch
    def update_mini_batch(self, mini_batch, mini_batchs_label, eta):
        # 保存每层的偏导
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 训练每一个mini_batch
        for x, y in zip(mini_batch, mini_batchs_label):
            delta_nabla_b, delta_nabla_w = self.update(x, y)

            # 保存一次训练网络中每层的偏导
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # 更新权重和偏执 Wn+1 = wn - eta * nw
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    # 前向传播
    def update(self, x, y):
        # 保存每层偏导
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x

        # 保存每一层的激励值a=sigmoid(z)
        activations = [x]

        # 保存每一层的z=wx + b
        zs = []
        # 前向传播
        for b, w in zip(self.biases, self.weights):
            # 计算每层的z
            z = np.dot(w, activation) + b

            # 保存每层的z
            zs.append(z)

            # 计算每层的a
            activation = sigmoid(z)

            # 保存每层的a
            activations.append(activation)

        # 反向更新
        # 计算最后一层的误差
        delta = self.cost_dericative(activations[-1], y) * sigmoid_prime(zs[-1])

        # 最后一层权重和偏置的导数
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # 导数第二层一直到第一层 权重和偏执的导数
        for l in range(2, self.num_layers):
            z = zs[-l]

            sp = sigmoid_prime(z)

            # 当前层的误差
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp

            # 当前层的偏置和权重的导数
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data, test_labels):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in zip(test_data, test_labels)]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_dericative(self, output_activation, y):
        return (output_activation - y)

if __name__ == '__main__':
    from personal.test.py3env import MinistData

    train_data_set, train_labels = MinistData.get_training_data_set()
    test_data_set, test_labels = MinistData.get_test_data_set()

    net = Network([784, 30, 10])
    net.SGD(train_data_set, train_labels, 30, 10, 0.5, test_data=test_data_set, test_labels=test_labels)