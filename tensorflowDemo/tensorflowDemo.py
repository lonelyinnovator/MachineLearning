# -*- coding: utf-8 -*-
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


def tensorflowDemo():
    greeting = tf.constant('Hello Tensorflow!')
    sess = tf.compat.v1.Session()
    result = sess.run(greeting)
    print(result)
    sess.close()


if __name__ == '__main__':
    tensorflowDemo()
