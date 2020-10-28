"""
model.py is used to train the Convolution Neural Network on dataset.
"""
from .helper import read_data
import tensorflow as tf
from numpy.random import seed
seed(43)

# Initialize parameters
batch_size = 16
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
n_class = len(classes)
val_size = 0.25
img_size = 64
n_channels = 3
img_dir = "../data/"  # dataset for training directory

# data
session = tf.Session()
X = tf.placeholder(tf.float32, shape=[None, img_size, img_size, n_channels],
                   name='X')

# labels
y_true = tf.placeholder(tf.float32, shape=[None, n_class], name='y_true')
y_true_class = tf.argmax(y_true, dimension=1)

# save model
saver = tf.train.Saver()

# read the training images from its directory
data = read_data(img_dir, img_size, classes, val_size)

print("dataset loaded!!!!!! \n Total Number of split (train, validation) dataset")
print("({}, {})".format(len(data.train.labels), len(data.valid.labels)))

# create weight method
def create_filter(shape):
    """
    create filters(weights)
    :param shape: dimension of the filter
    :return: filter matrix (which is a tensorflow variable)
    """

    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

# create bias method
def create_bias(size):
    """
    create bias(b)
    :param size: its a constant variable
    :return: bias vector
    """
    return tf.Variable(tf.constant(0.05, shape=[size]))

# create layers
# convolution layer.
def create_conv_layer(input, n_channels, filter_size, n_filters):
    """
    create convolution layer including max pooling
    :param input:
    :param n_channels:
    :param filter_size:
    :param n_filters:
    :return: convoluted layer
    """
    # define convolution filter/weight
    filter = create_filter(shape=[filter_size, filter_size,
                                  n_channels, n_filters])
    # define bias
    bias = create_bias(n_filters)

    # define convolution layer
    layer = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
    layer += bias

    return layer


# create max pooling layer
def create_maxpool(layer):
    """
    max pooling operation is performed.
    :param layer: convolution layer (input)
    :return: max-pooled layer
    """
    return tf.nn.max_pool(layer, ksize=[1, 2, 2 ,1],
                          strides=[1, 2, 2, 1], padding='SAME')


# create activation function
def create_activation(layer):
    """
    perform 'relu' activation operation
    :param layer: can be any layer
    :return: relu performed layer (matrix elements are in 0 or 1)
    """
    return tf.nn.relu(layer)

# create flatten layer
def create_flat(layer):
    """
    convert convolution layer into flatten output.
    :param layer:
    :return: flatten layer
    """
    layer_shape = layer.get_shape()
    n_features = layer_shape[1: 4].n_elements() # num_elements()
    return tf.reshape(layer, shape=[-1, n_features]), n_features

# create dense layer
def create_dense(layer, n_inputs, n_outputs, use_relu = True):
    """
    create fully connected layer
    :param layer: flatten layer as input
    :param n_inputs: number of inputs
    :param n_outputs: number of outputs
    :param use_relu: going to use relu activation set to always True
    :return: fully connected layer
    """
    weights = create_filter(shape=[n_inputs, n_outputs])
    bias = create_bias(n_outputs)
    #  X*weights + b
    layer = tf.matmul(layer, weights) + bias

    if use_relu:
        layer = tf.nn.relu(layer)
    return layer

class Model:
    """
    training model
    """
    def __init__(self, X):
        # layer 1
        self.conv1 = create_conv_layer(X, 3, 2, 32)
        self.maxpool1 = create_maxpool(self.conv1)
        self.relu1 = create_activation(self.maxpool1)
        # layer 2
        self.conv2 = create_conv_layer(self.relu1, 32, 2, 32)
        self.maxpool2 = create_maxpool(self.conv2)
        self.relu2 = create_activation(self.maxpool2)
        # layer 3
        self.conv3 = create_conv_layer(self.relu2, 32, 2, 64)
        self.maxpool3 = create_maxpool(self.conv3)
        self.relu3 = create_activation(self.maxpool3)
        # layer 4
        self.conv4 = create_conv_layer(self.relu3, 64, 2, 32)
        self.maxpool4 = create_maxpool(self.conv4)
        self.relu4 = create_activation(self.relu4)
        # layer 5
        self.conv5 = create_conv_layer(self.relu4, 32, 2, 64)
        self.maxpool5 = create_maxpool(self.conv5)
        self.relu5 = create_activation(self.maxpool5)
        # flat layer
        self.flat, self.n_inputs = create_flat(self.relu5)
        # fully connected layer 1
        self.dense1 = create_dense(self.flat,self.n_inputs, 1024)
        # fully connected layer 2
        self.dense2 = create_dense(self.dense1, 1024, n_class, use_relu=False)
        self.accuracy, self.optimizer, self.cost = self.prediction()

        def prediction():
            """
            predicts the output
            :return: predicted output
            """
            y = tf.nn.softmax(self.dense2, name='y')
            y_class = tf.argmax(y, dimension=1)
            session.run(tf.global_variabls_initializer())
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.dense2,
                                                                    labels=y_true)
            cost = tf.reduce_mean(cross_entropy)
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
            correct_prediction = tf.equal(y_class, y_true_class)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            session.run(tf.global_variables_initializer())
            return accuracy, optimizer, cost

        def show_verbose(epoch, feed_trainset, feed_valset, val_loss):
            """
            verbose gives details about ongoing training
            :param epoch: number of iteration
            :param feed_trainset: training dataset
            :param feed_valset: validation dataset
            :param val_loss: validation loss
            :return: detail message about training
            """
            train_accuracy = session.run(self.accuracy, feed_dict=feed_trainset)
            val_accuracy = session.run(self.accuracy, feed_dict=feed_valset)
            msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
            print(msg.format(epoch + 1, train_accuracy, val_accuracy, val_loss))




if __name__=="__main__":
    """
    main function.
    """
    # iterations
    total_iterations = 0
    n_iterations = 1000

    model = Model(X)
    for i in range(total_iterations, total_iterations + n_iterations):
        X_batch, y_true_batch, _, class_batch = data.train.next_batch(batch_size)
        X_val_batch, y_val_batch, _, val_class_batch = data.valid.next_batch(batch_size)

        feed_dict_train = {X: X_batch, y_true : y_true_batch}
        feed_dict_val = {X: X_val_batch, y_true: y_val_batch}

        session.run(model.optimizer, feed_dict=feed_dict_train)

        if i % int(data.train.n_samples / batch_size) == 0:
            val_loss = session.run(model.cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.n_examples / batch_size))
            model.show_verbose(epoch, feed_dict_train, feed_dict_val, val_loss)
            saver.save(session, "../model/train_model")
    total_iterations += n_iterations


