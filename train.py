import argparse
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import gender_model as gender

np.set_printoptions(threshold=np.nan)
np.random.seed = 42


def load_data(classifier, samples, samples_test):
    x_train = np.load(classifier + "-features-train.npy")
    y_train = np.load(classifier + "-classes-train.npy")
    x_test = np.load(classifier + "-features-test.npy")
    y_test = np.load(classifier + "-classes-test.npy")
    return x_train[:samples], y_train[:samples], x_test[:samples_test], y_test[:samples]


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--steps", type=int, default=100)
    FLAGS, unknown = parser.parse_known_args()
    samples = FLAGS.samples
    steps = FLAGS.steps
    train_summary_dir = './logs/1'
    gender.train(train_summary_dir, steps, samples)


if __name__ == "__main__":
    main()
