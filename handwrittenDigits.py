############################################################################################################################
# TODO:

# 1. Download MNIST Dataset
# 2. Setup Neural Network with required number of layers & nodes and then train it
# 3. Check output for 1 image
# 4. Feed the Test Dataset to the trained Neural Network & check its accuracy


############################################################################################################################
import urllib
import gzip
import os
import numpy as np

def load_dataset():
    print "********** Started Loading Data **********"

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        # print "********** Started Downloading Data **********"
        urllib.urlretrieve(source+filename,  filename)
        # print "********** Completed Downloading Data **********"

    def load_mnist_images(filename):
        if not (os.path.exists(filename)):
            download(filename)

        with gzip.open(filename, 'rb') as f:
            # print f
            data = np.frombuffer(f.read(), np.uint8, offset=16)
            # print data
            data = data.reshape(-1,1,28,28)
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

    print "********** Completed Loading Data **********"
    return X_train, Y_train, X_test, Y_test


X_train, Y_train, X_test, Y_test = load_dataset()

