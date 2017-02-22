import tensorflow as tf
import numpy as np
from PIL import Image
import base64
from io import BytesIO


class MnistPredictor(object):
    def __init__(self, model_path='./utils/saved_models/model.ckpt'):

        def weight_variable(shape, name):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial, name=name)

        def bias_variable(shape, name):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial, name=name)

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Input layer
        self.x = tf.placeholder(tf.float32, [None, 784], name='x')
        self.y_ = tf.placeholder(tf.float32, [None, 10],  name='y_')
        x_image = tf.reshape(self.x, [-1, 28, 28, 1])

        # Convolutional layer 1
        W_conv1 = weight_variable([5, 5, 1, 32], "W_conv1")
        b_conv1 = bias_variable([32], "b_conv1")

        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        # Convolutional layer 2
        W_conv2 = weight_variable([5, 5, 32, 64], "W_conv2")
        b_conv2 = bias_variable([64], "b_conv2")

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        # Fully connected layer 1
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

        W_fc1 = weight_variable([7 * 7 * 64, 1024], "W_fc1")
        b_fc1 = bias_variable([1024], "b_fc1")

        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Dropout
        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # Fully connected layer 2 (Output layer)
        W_fc2 = weight_variable([1024, 10], "W_fc2")
        b_fc2 = bias_variable([10], "b_fc2")

        self.y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(init_op)
        saver.restore(self.sess, model_path)

    def train_model(self, batches=20000):
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("train_data/", one_hot=True)
        for i in range(batches):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = self.accuracy.eval(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0})
                print("step %d, training accuracy %g" % (i, train_accuracy))
            self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})
        saver = tf.train.Saver()
        print("Model saved in file: %s" % saver.save(self.sess, "./utils/saved_models/model.ckpt"))

    def get_prediction(self, uri):
        img = Image.open(BytesIO(base64.b64decode(uri[22:])))
        img = self.remove_transparency(img)
        img.thumbnail((28, 28))
        img = img.convert("L")
        data = np.asarray(img, dtype="float32")
        flattened_data = 1 - (data.flatten()) / 255
        input_list = [flattened_data]
        guess_prob, guessed = self.sess.run([self.y_conv, tf.argmax(self.y_conv, 1)], feed_dict={self.x: input_list, self.keep_prob: 1.0})
        return str(guessed[0])

    def remove_transparency(self, im, bg_colour=(255, 255, 255)):
        if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
            alpha = im.convert('RGBA').split()[-1]
            bg = Image.new("RGBA", im.size, bg_colour + (255,))
            bg.paste(im, mask=alpha)
            return bg
        else:
            return im