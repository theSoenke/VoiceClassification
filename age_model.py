import os
import numpy as np
from keras.utils import np_utils
import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator
import train as train_model


def train(summary_dir, steps, samples):
    time_steps = 128
    num_classes = 8
    feature_size = 13
    learning_rate = 0.001
    training_steps = steps

    process_id = os.getenv('SLURM_PROCID')
    if process_id == 1:
        learning_rate = 0.01
    print("Learning rate: " + str(learning_rate))

    x_train, y_train, x_test, y_test = train_model.load_data("age", samples, 500)
    y_train = tf.one_hot(y_train, num_classes)
    x_test = x_test.reshape((-1, time_steps, feature_size))
    y_test = tf.one_hot(y_test, num_classes)

    x, y, loss, accuracy, optimizer, summary_op = train_model.build_graph(feature_size, time_steps, num_classes, learning_rate)
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        for step in range(1, training_steps + 1):
            total_loss = 0.0
            total_accuracy = 0.0
            total_steps = 0
            iterator = Iterator.from_structure(train_data.output_types, train_data.output_shapes)
            next_element = iterator.get_next()
            sess.run(iterator.make_initializer(train_data))
            print("Step: ", step)
            while True:
                try:
                    total_steps += 1
                    x_element, y_element = sess.run(next_element)
                    x_element = x_element.reshape((1, time_steps, feature_size))
                    y_element = y_element.reshape((-1, num_classes))
                    feed_dict = {x: x_element, y: y_element}
                    loss_value, accuracy_value, _ = sess.run([loss, accuracy, optimizer], feed_dict=feed_dict)
                    total_loss += loss_value
                    total_accuracy += accuracy_value
                except tf.errors.OutOfRangeError:
                    break

            if step % 5 == 0:
                feed_dict = {x: x_train, y: y_train.eval()}
                _, _, summary = sess.run([loss, accuracy, summary_op], feed_dict=feed_dict)
                summary_writer.add_summary(summary, step)

            print("Loss: ", total_loss / total_steps)
            print("Accuracy: ", total_accuracy / total_steps)

        saver.save(sess, './models/model-age.ckpt')

        print("Train Accuracy: ", sess.run(accuracy, feed_dict={x: x_train, y: y_train.eval()}))
        # print("Test Accuracy: ", sess.run(accuracy, feed_dict={x: x_test, y: y_test.eval()}))
