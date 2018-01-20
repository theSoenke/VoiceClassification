import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from keras.utils import np_utils
from tensorflow.contrib.data import Dataset, Iterator

np.set_printoptions(threshold=np.nan)


def load_data():
    x_train = np.load("features-train.npy")
    y_train = np.load("classes-train.npy")
    x_test = np.load("features-test.npy")
    y_test = np.load("classes-test.npy")
    return x_train, y_train, x_test, y_test


def RNN(x, time_steps, num_hidden, num_classes):
    x = tf.unstack(x, time_steps, 1)
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    outputs, hidden_states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    w = tf.Variable(tf.truncated_normal([num_hidden, num_classes]))
    b = tf.Variable(tf.truncated_normal([num_classes]))

    return tf.matmul(outputs[-1], w) + b


def build_graph(feature_size, time_steps, num_classes, learning_rate):
    num_hidden = 128
    x = tf.placeholder("float", [None, time_steps, feature_size])
    y = tf.placeholder("float", [None, num_classes])

    logits = RNN(x, time_steps, num_hidden, num_classes)
    prediction = tf.nn.softmax(logits)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    tf.summary.histogram("histogram loss", loss)
    summary_op = tf.summary.merge_all()

    return x, y, loss, accuracy, optimizer, summary_op


def main():
    time_steps = 128
    num_classes = 2
    feature_size = 33
    learning_rate = 0.001
    training_steps = 100
    x_train, y_train, x_test, y_test = load_data()
    y_train = np_utils.to_categorical(y_train)
    x, y, loss, accuracy, optimizer, summary_op = build_graph(feature_size, time_steps, num_classes, learning_rate)
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    with tf.Session() as sess:
        train_summary_dir = '/tmp/commonvoice/1'
        summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        for step in range(1, training_steps + 1):
            total_loss = 0.0
            iterator = Iterator.from_structure(train_data.output_types, train_data.output_shapes)
            next_element = iterator.get_next()
            sess.run(iterator.make_initializer(train_data))
            print("Step: ", step)
            while True:
                try:
                    x_element, y_element = sess.run(next_element)
                    x_element = x_element.reshape((1, time_steps, feature_size))
                    y_element = y_element.reshape((-1, 2))
                    feed_dict = {x: x_element, y: y_element}
                    loss_value, acc, _, summary= sess.run([loss, accuracy, optimizer, summary_op], feed_dict=feed_dict)
                    summary_writer.add_summary(summary, step)
                    total_loss += loss_value
                except tf.errors.OutOfRangeError:
                    break

            loss_avg = total_loss / (training_steps + 1)
            print("Loss Average: ", loss_avg)

        saver.save(sess, './model.ckpt')
        # x_test = x_test.reshape((-1,time_steps, feature_size))
        # y_test = y_test.reshape((-1,2))
        # print("Test Accuracy: ", sess.run(accuracy, feed_dict={x: x_test, y: y_test}))


if __name__ == "__main__":
    main()
