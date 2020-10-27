"""
helper.py is a dataset loader function which loads the dataset
"""

## Library
import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np

class dataset(object):
    """
    dataset class
    """

    def __init__(self, images, labels, img_names, cls):
        """
        constructor function
        :param images: in numpy format float 32
        :param labels: image labels
        :param img_names: image names
        :param cls: image class
        """
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._img_names = img_names
        self._cls = cls
        self._epochs_done = 0
        self._index_in_epoch = 0

        @property
        def images(self):
            return self._images

        @property
        def labels(self):
            return self._labels

        @property
        def img_names(self):
            return self._img_names

        @property
        def cls(self):
            return self._cls

        @property
        def num_examples(self):
            return self._num_examples

        @property
        def epochs_done(self):
            return self._epochs_done

    def next_batch(self, batch_size):
        """
        returns next batch of images and image associated values from the dataset
        :param batch_size:  the amount of files to be loaded.
        :return: _images[start:end], _labels[start:end], _img_names[start:end], _cls[start:end]
        """
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # After each epoch we update this
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]

def load_train(path, img_size, classes):
    """
    load_train function reads the dataset and returns images, image labels, image names and images class
    :param path: directory where dataset is located
    :param img_size: dimensions of images (e.g  512*512*3)
    :param classes: dis
    :return: list of images, labels, img_names, cls
    """
    images = []
    labels = []
    img_names = []
    cls = []

    print("Reading training images..........................")

    for fields in classes:
        idx = classes.index(fields)
        print("Reading {} files (Index: {}) ".format(fields, idx))
        path = os.path.join(path, fields, '*g')
        files = glob.glob(path)
        for file in files:
            # image data engineering....
            img = cv2.imread(file)
            img = cv2.resize(img, (img_size, img_size), 0, 0, cv2.INTER_LINEAR)
            img = img.astype(np.float32)
            img = np.multiply(img, 1.0/255.0)
            images.append(img)

            # image labels data engineering...
            lbl = np.zeros(len(classes))
            lbl[idx] = 1.0
            labels.append(lbl)

            # image names appending in img_names list
            filebase = os.path.basename(file)
            img_names.append(filebase)

            # append fields into cls list
            cls.append(fields)

    return images, labels, img_names, cls


def read_data(path, img_size, cls, val_size):
    """
    read the training dataset and validation set
    :param path: directory of training dataset
    :param img_size: image dimensions
    :param cls: image class
    :param val_size: validation dataset size
    :return: python class name data
    """
    class datasets(object):
        """
        empty pthon class
        """
        pass

    data = datasets()

    images, labels, img_names, cls = load_train(path, img_size, cls)
    images, labels, img_names, cls = shuffle(images, labels, img_names, cls)

    if isinstance(val_size, float):
        validation_size = int(val_size * images.shape[0])

    val_imgs = images[:val_size]
    val_lbls = labels[:val_size]
    val_img_names = img_names[:val_size]
    val_cls = cls[:val_size]

    train_imgs = images[val_size:]
    train_lbls = labels[val_size:]
    train_img_names = img_names[val_size:]
    train_cls = cls[val_size:]

    data.train = dataset(train_imgs, train_lbls, train_img_names, train_cls)
    data.valid = dataset(val_imgs, val_lbls, val_img_names, val_cls)

    return data