import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

points_num = 100
vectors = []

for i in range(points_num):
    x1 = np.random.normal(0.0, 0.6)
    y1 = max(0.1 * x1 + 0.2 + np.random.normal(0.0, 0.1), 0)
    vectors.append([x1, y1])

x_data = [v[0] for v in vectors]
y_data = [v[1] for v in vectors]

w = tf.Variable(tf.random_uniform([1], -1.0, 1.0), dtype=tf.float32, name='weight')
b = tf.Variable(tf.zeros([1]), dtype=tf.float32, name='bias')
with tf.name_scope("Output"):
    y = tf.nn.relu(w * x_data + b)
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.3) # 设置学习率为 0.5
train = optimizer.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

y_res = []
for step in range(1000):
    sess.run(train)
    y_res.append(sess.run(y))

path = "./log"

fig, ax = plt.subplots()
line, = ax.plot(x_data, y_data)

def init():  # only required for blitting to give a clean slate.
    line.set_ydata(y_res[0])
    return line,


def animate_func(i):
    print(y_res[(i%10000)])
    line.set_ydata(y_res[(i % 10000)])  # update the data.
    return line,


ani = animation.FuncAnimation(
    fig, animate_func, init_func=init, interval=100, blit=True, save_count=50)

# To save the animation, use e.g.
#
# ani.save("movie.html")
plt.plot(x_data, y_data, 'r*', label="Original data")
plt.show()