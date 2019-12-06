#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle as pk


#%%
# 获取数据
with open('./x_train_all.pk', 'rb') as fp: x_data_all = np.array(pk.load(fp))
with open('./y_train_all_31.pk', 'rb') as fp: y_data_all = np.array(pk.load(fp))

# 数据分箱
x_train_all, x_test, y_train_all, y_test = train_test_split(
    x_data_all, y_data_all, random_state=7
)
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_all, y_train_all, random_state=11
)
# 数据归一化
scaler = StandardScaler()
x_train_scaler = scaler.fit_transform(x_train)
x_valid_scaler = scaler.transform(x_valid)
x_test_scaler = scaler.transform(x_test)

#%%
# 加载模型
load_type = 1
if load_type == 1:
    model = tf.keras.models.load_model(os.path.join('callbacks', 'loss_user_mnist.h5'))
print(model.summary())
# 测试模型准度
loss, acu = model.evaluate(x_test_scaler, y_test)
print(loss, acu)

#%%
# 建立模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(30, activation='relu', input_shape=x_test_scaler.shape[1:]),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])
# 编译模型
model.compile(loss='sparse_categorical_crossentropy',
              optimizer = 'sgd',
              metrics = ['accuracy']
             )
# 设置回调函数
logdir = '.\\callbacks'

if not os.path.exists(logdir):
    os.mkdir(logdir)

output_model_file = os.path.join(logdir, 'loss_user_mnist.h5')

callbacks = [
    tf.keras.callbacks.TensorBoard(logdir),
    tf.keras.callbacks.ModelCheckpoint(output_model_file, save_best_only = True),
    tf.keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3),
]
history = model.fit(x_train_scaler, y_train, epochs=100,
                    validation_data=(x_valid_scaler, y_valid),
                   callbacks=callbacks)

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(16, 8))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()
plot_learning_curves(history)
