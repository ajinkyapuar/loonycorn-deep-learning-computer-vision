############################################################################################################################
# TODO:

# 1. Download MNIST Dataset
# 2. Setup Neural Network with required number of layers & nodes and then train it
# 3. Check output for 1 image
# 4. Feed the Test Dataset to the trained Neural Network & check its accuracy

# Higher dimensional array are often called tensors. We use Theano to work with tensors.
# Lasagne is built on top of Theano and is used to build Neural Networks.
############################################################################################################################
import urllib
import gzip
import os
import numpy as np
# import matplotlib.pyplot as plt

import theano
import theano.tensor as T
import lasagne


############################################################################################################################

def load_dataset():
    # print "********** Started Loading Data **********"

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        # print "********** Started Downloading Data **********"
        urllib.urlretrieve(source + filename, filename)
        # print "********** Completed Downloading Data **********"

    def load_mnist_images(filename):
        if not (os.path.exists(filename)):
            download(filename)

        with gzip.open(filename, 'rb') as f:
            # print f
            data = np.frombuffer(f.read(), np.uint8, offset=16)
            # print data
            data = data.reshape(-1, 1, 28, 28)
            return data / np.float32(256)

    def load_mnist_labels(filename):
        if not (os.path.exists(filename)):
            download(filename)

        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
            return data

    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    Y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    Y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')
    # print "********** Completed Loading Data **********"
    return X_train, Y_train, X_test, Y_test


X_train, Y_train, X_test, Y_test = load_dataset()


# plt.show(plt.imshow(X_train[1][0]))

############################################################################################################################

def build_NN(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)

    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    l_hid1 = lasagne.layers.DenseLayer(l_in_drop, num_units=800, nonlinearity=lasagne.nonlinearities.rectify,
                                       W=lasagne.init.GlorotUniform())  # weights initialization

    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    l_hid2 = lasagne.layers.DenseLayer(l_hid1_drop, num_units=800, nonlinearity=lasagne.nonlinearities.rectify,
                                       W=lasagne.init.GlorotUniform())

    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

    l_out = lasagne.layers.DenseLayer(l_hid2_drop, num_units=10, nonlinearity=lasagne.nonlinearities.softmax)

    return l_out


input_var = T.tensor4('inputs')
target_var = T.ivector('targets')

network = build_NN(input_var)

prediction = lasagne.layers.get_output(network)

loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)

loss = loss.mean()

params = lasagne.layers.get_all_params(network, trainable=True)

updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)

train_fn = theano.function([input_var, target_var], loss, updates=updates)

############################################################################################################################

num_training_steps = 10

for step in range(num_training_steps):
    train_err = train_fn(X_train, Y_train)
    print "Current Step: " + str(step)

############################################################################################################################

test_prediction = lasagne.layers.get_output(network)

val_fn = theano.function([input_var], test_prediction)

scores_array = val_fn([X_test[5]])

print scores_array

print "Max Score: " + str(max(scores_array[0]))

print Y_test[5]

############################################################################################################################

test_prediction = lasagne.layers.get_output(network, deterministic=True)

test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

acc_fn = theano.function([input_var, target_var], test_acc)

print acc_fn(X_test, Y_test)
