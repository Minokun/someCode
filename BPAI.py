import numpy as np
from personal.test.py3env.MinistData import *
from datetime import datetime

# 全连接层实现类
class FullConnectedLayer():
    def __init__(self, input_size, output_size, activator):
        '''
        构造函数
        input_size: 本层输入向量的纬度
        output_size: 本层输出向量的纬度
        activator: 激活函数
        '''

        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        # 权重数组W
        self.W = np.random.uniform(-0.1, 0.1, (output_size, input_size))

        # 偏置项b
        self.b = np.zeros((output_size, 1))

        # 输出项
        self.output = np.zeros((output_size, 1))

    def forward(self, input_array):
        '''
        前向计算
        input_array: 输入向量，纬度必须等于input_size
        '''
        self.input = input_array
        self.outpu = self.activator.forward(
            np.dot(self.W, input_array) + self.b)

    def backward(self, delta_array):
        '''
        反向计算W和B的梯度
        delta_array: 从上一层传递过来的误差项
        '''
        delta = self.activator.backward(self.input) * np.dot(self.W.T, delta_array)
        self.W_grad = np.dot(delta, self.W.T)
        self.b_grad = delta

    def update(self, learning_rate):
        '''
        梯度下降算法更新权重
        '''
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad


# 激活函数
class SigmoidActivator():
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        return output * (1 - output)


# 神经网络类
class Network():
    def __init__(self, layers):
        '''
        构造函数
        '''
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(
                FullConnectedLayer(
                    layers[i], layers[i + 1],
                    SigmoidActivator()
                )
            )

    def predict(self, sample):
        '''
        使用神经网络实现预测
        sample: 输入样本
        '''
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def train(self, labels, data_set, rate, epoch):
        '''
        训练函数
        labels: 样本标签
        data_set: 输入样本
        rate: 学习速率
        epoch: 训练轮数
        '''
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)

    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)

    def calc_gradient(self, label):
        delta = self.layers[-1].activator.backward(self.layers[-1].output
                                                   ) * (label - self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)


def get_result(vec):
    max_value_index = 0
    max_value = 0
    for i in range(len(vec)):
        if vec[i] > max_value:
            max_value = vec[i]
            max_value_index = i
    return max_value_index


def evaluate(network, test_data_set, test_labels):
    error = 0
    total = len(test_data_set)
    for i in range(total):
        label = get_result(test_labels[i])
        predict = get_result(network.predict(test_data_set[i]))
        if label != predict:
            error += 1
    return float(error) / float(total)


def train_and_evaluate():
    last_error_ratio = 1.0
    epoch = 0
    train_data_set, train_labels = get_training_data_set()
    test_data_set, test_labels = get_test_data_set()
    network = Network([784, 300, 10])
    while True:
        epoch += 1
        network.train(train_labels, train_data_set, 0.3, 1)
        print('%s epoch %d finished' % (datetime.now(), epoch))
        if epoch % 10 == 0:
            error_ratio = evaluate(network, test_data_set, test_labels)
            print('%s after epoch %d, error ratio is %f' % (datetime.now(), epoch, error_ratio))
            if error_ratio > last_error_ratio:
                break
            else:
                last_error_ratio = error_ratio


train_and_evaluate()