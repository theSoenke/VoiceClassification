import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from keras.utils import np_utils
from tensorflow.contrib.data import Dataset, Iterator


def load_data():
    x_train = np.load("features-train.npy")
    y_train = np.load("classes-train.npy")
    return x_train, y_train


def RNN(x, time_steps, num_hidden, num_classes):
    x = tf.unstack(x, time_steps, 1)
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    outputs, hidden_states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    w = tf.Variable(tf.random_normal([num_hidden, num_classes]))
    b = tf.Variable(tf.random_normal([num_classes]))

    return tf.matmul(outputs[-1], w) + b


def build_graph(feature_size, time_steps, num_classes, learning_rate):
    num_hidden = 128
    x = tf.placeholder("float", [None, time_steps, feature_size])
    y = tf.placeholder("float", [None, num_classes])

    logits = RNN(x, time_steps, num_hidden, num_classes)
    prediction = tf.nn.softmax(logits)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return x, y, loss, accuracy, optimizer


def main():
    time_steps = 128
    num_classes = 2
    feature_size = 33
    learning_rate = 0.001
    training_steps = 100
    display_step = 100
    x_train, y_train = load_data()
    y_train = np_utils.to_categorical(y_train)
    x, y, loss, accuracy, optimizer = build_graph(feature_size, time_steps, num_classes, learning_rate)
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(1, training_steps + 1):
            losses = []
            iterator = Iterator.from_structure(train_data.output_types, train_data.output_shapes)
            next_element = iterator.get_next()
            training_init_op = iterator.make_initializer(train_data)
            sess.run(training_init_op)
            print("Step: ", step)
            while True:
                try:
                    x_element, y_element = sess.run(next_element)
                    x_element = x_element.reshape((1, time_steps, feature_size))
                    y_element = y_element.reshape((-1, 2))
                    sess.run([loss, accuracy, optimizer], feed_dict={x: x_element, y: y_element})
                    loss_value, acc = sess.run([loss, accuracy], feed_dict={x: x_element, y: y_element})
                    losses.append(loss_value)
                except tf.errors.OutOfRangeError:
                    break

            loss_sum = np.sum(np.array(losses))
            loss_avg = loss_sum / len(losses)
            print("Loss Average: ", loss_avg)


if __name__ == "__main__":
    main()
