# https://pyimagesearch.com/2016/10/10/gradient-descent-with-python/?_ga=2.126683149.1280261111.1703145897-1842902230.1698424416
# https://www.mldawn.com/deriving-the-gradient-descent-rule-part-1/
# https://towardsdatascience.com/gradient-descent-show-me-the-math-7ba7d1caef09

# All gradient descent methods contain the follow step
# while True: # some condition needs to be met for the gradient descent to terminate
#     W_gradient = evaluate_gradient(loss, data, W)
#     W += -alpha * W_gradient # alpha is the learning rate
#
# The potential conditions to terminate gradient descent include:
# 1. a specified number of iterations have been completed
# 2. the loss has become sufficiently low or training accuracy sufficiently high
# 3. the loss has not improved after a certain number of iterations
#
# The direction of the "gradient" is technically the direction of the gradient of the loss function
# However, since we do not have any better basis for iteratively solving for the best W (weight),
# we also use this gradient to iteratively approach the optimal W (weight)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

# idea:
# we are trying to train a neural network with 1 layer, with inputs with 2 features, 
# x0, x1, and a 2x1 weights matrix (2 features, predicts 1 class after dot product)
# The only node in this neural network is one that performs sigmoid activation
# This simple neural network is good for performing a slimple binary classification
# expected input is an array of values. By doing element-wise computation, the
# activation value of each data point can be calculated
# expecte output has shape (n, 1), where n represents the number of data points,
# and 1 represents the output of the activation function for one data point
def sigmoid_activation(x): 
    # The sigmoid activation function
    return 1.0 / (1 + np.exp(-x))

# expected input is an array of values. By doing element-wise computation, the
# activation value of each data point can be calculated
# expecte output has shape (n, 1), where n represents the number of data points,
# and 1 represents the output of the activation function for one data point
def sigmoid_derivative(x):
    # the derivative of the sigmoid function
    # This derivative was calculated deterministically through calculus
    return sigmoid_activation(x) * (1 - sigmoid_activation(x))

def predict(X, W):
    # take the dot product between the data points and weight matrix
    predictions = sigmoid_activation(X.dot(W))

    # step function:
    # if the output of the sigmoid function > 0.5, then assign class label 1
    # otherwise, assign class label 0
    predictions[predictions <= 0.5] = 0
    predictions[predictions > 0.5] = 1

    return predictions

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--iterations", type=float, default=100, help="number of iterations")
ap.add_argument("-a", "--alpha", type=float, default=0.01, help="learning rate alpha")
args = vars(ap.parse_args())

# make_blobs:
# randomly generates datapoints
# inputs:
# n_samples: the number of data points
# n_features: the number of features each data point has. For instance, for n_features=2, the data points will look something like [x0, x1], [x2, x3], ...
# centers: essentially, how many "classes" there are; literally, how many centers data should cluster around
# cluster_std: on average, how far away data points should be from their centers
# random_state: kind of like seeding. Used to make sure that the random results are reproducible
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
# X.shape = (1000, 2) 2-D array
# y.shape = (1000, ) 1-D array
y = y.reshape((y.shape[0], 1))
# y.shape = (1000, 1)

# append a 3rd "feature column" of ones in X to represent the bias that will be calculated when dotting with the W (weights) matrix
X = np.c_[X, np.ones((X.shape[0]))]
# X.shape = (1000, 3), 1000 samples, each having 3 features (one feature is 1, for the bias in the weight matrix)

(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

W = np.random.rand(X.shape[1], 1)
# W.shape = (3, 1), including the bias term
losses = []

for iteration in np.arange(0, args["iterations"]):
    # the input to the activation function: X.dot(W) <- dot product between the data and the weights
    # expected output has shape (n, 1), where n is the number of data points, and 1 represents the value input to the activation function
    # The prediction values here are not binary
    predictions = sigmoid_activation(trainX.dot(W))
    error = predictions - trainY
    # squared loss function
    loss = np.sum(error ** 2)
    losses.append(loss)

    # determined through calculus
    d = error * sigmoid_derivative(trainX.dot(W))
    gradient = trainX.T.dot(d)

    W = W - args["alpha"] * gradient
    if iteration == 0 or (iteration + 1) % 5 == 0:
	    print(f"[INFO] iteration={iteration + 1}, loss={loss}")

test_case_predictions = predict(testX, W)
test_case_predictions = test_case_predictions.astype("int")
print(classification_report(testY, test_case_predictions))

plt.style.use("ggplot")
plt.figure()
plt.title("data")
plt.scatter(testX[:, 0], testX[:, 1], marker='o', c=testY[:, 0], s=30)

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["iterations"]), losses)
plt.title("training loss")
plt.xlabel("# of iterations")
plt.ylabel("loss")
plt.show()