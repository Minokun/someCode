#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import os


# In[2]:


import pickle as pk
with open('./x_train_all.pk', 'rb') as fp: user_behavior_data = np.array(pk.load(fp))
with open('./y_train_all.pk', 'rb') as fp: user_behavior_target = np.array(pk.load(fp))


# In[3]:


x_train_all, x_test, y_train_all, y_test = train_test_split(
    user_behavior_data, user_behavior_target, random_state = 7
)
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_all, y_train_all, random_state = 11
)


# In[4]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled =  scaler.transform(x_test)


# In[5]:


# 三层网络 sparse_categorical_crossentropy
model = keras.models.Sequential([
    keras.layers.Dense(600, activation='relu', input_shape=x_train_scaled.shape[1:]),
    keras.layers.Dense(2, activation='softmax')
])
model.compile(loss='mean_squared_error',
              optimizer = 'sgd',
              metrics = ['accuracy']
             )

tf.losses
# In[23]:


# 深度神经网络建立
model = keras.models.Sequential()

model.add(keras.layers.Dense(12, input_shape=x_train_scaled.shape[1:]))
for i in range(20):
    model.add(keras.layers.Dense(300, activation='selu'))
    if i % 5 == 1:
        model.add(keras.layers.AlphaDropout(rate=0.5))  
model.add(keras.layers.Dense(2, activation='softmax'))
    
model.compile(loss='sparse_categorical_crossentropy',
              optimizer = 'sgd',
              metrics = ['accuracy']
             )


# In[6]:


model.summary()


# In[30]:


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(16, 8))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()
plot_learning_curves(history)


# In[7]:


logdir = '.\callbacks'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir, 'loss_user_mnist.h5')

callbacks = [
    keras.callbacks.TensorBoard(logdir),
    keras.callbacks.ModelCheckpoint(
        output_model_file, 
        save_best_only = True),
    keras.callbacks.EarlyStopping(patience=5,
                                 min_delta=1e-3),
]
history = model.fit(x_train_scaled, y_train, 
                    epochs=100, 
                    validation_data=(x_valid_scaled, y_valid),
                   callbacks=callbacks)

