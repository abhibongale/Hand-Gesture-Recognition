"""
evaluate.py predicts the model performance on unseen/test dataset
"""
from glob import glob
import cv2
import numpy as np
import tensorflow as tf

# load the train model
session = tf.Session()
saver = tf.train.import_meta_graph('../data/train_model.meta')
saver.restore(session, tf.train.latest_checkpoint('../model/'))
graph = tf.get_default_graph()

y = graph.get_tensor_by_name("y:0")
X = graph.get_tensor_by_name("X:0")
y_true = graph.get_tensor_by_name("y_true:0")
y_test_imgs = np.zeros((1, 10))


def evaluate(path):
    """
    It test model on test dataset(unknown dataset)
    :param path: test dataset path
    :return:
    """
    img_size = 50
    n_channels = 3
    images = []

    for i in glob(path + "/*"):
        print(i)
        filename = i
        # read the images
        image = cv2.imread(filename)
        cv2.imshow('evaluate', image)

        # preprocessing dataset
        image = cv2.resize(image, (img_size, img_size), 0, 0, cv2.INTER_LINEAR)
        images.append(image)
        images = np.array(images, dtype=np.uint8)
        images = images.astype('float32')
        images = np.multiply(images, 1.0 / 255)

        # the input dimensions [None, img_size, img_size, n_channels]
        X_batch = images.reshape(1, img_size, img_size, n_channels)
        feed_dict_test = {X: X_batch, y_true: y_test_imgs}
        result = session.run(y, feed_dict=feed_dict_test)
        np.set_printoptions(formatter={'float_kind': '{:f}'.format})
        print(result)
        print(np.argmax(max(result)))
        k = cv2.waitKey()
        if k == 99:
            continue
