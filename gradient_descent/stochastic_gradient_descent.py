from vanilla_gradient_descent import sigmoid_activation, sigmoid_derivative, predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

def next_batch(X, y, batch_size):
    # The idea:
    # yield a subset of X and y at each iteration
    # for instance, if X.shape = (500, 3), y.shape = (500, 1), batch_size = 32,
    # then i = 0, 32, 64, 96, ... 480
    # and the batches will contain:
    # i = 0, X[0: 32], y[0:32]
    # i = 32, X[32:64], y[32:64]
    # ...
    # i = 480, X[480:512], y[480, 512]
    # even though 512 is out of bounds, no error will be thrown. effectively, it will be like X[480:] and y[480:]
    for i in np.arange(0, X.shape[0], batch_size):
        yield (X[i:i + batch_size], y[i:i + batch_size])
    

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--iterations", type=int, default=100, help="# of iterations")
ap.add_argument("-a", "--alpha", type=float, default=0.01, help="learning rate")
ap.add_argument("-b", "--batch_size", type=int, default=32, help="size of batch for stochastic gradient descent")
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

W = np.random.randn(X.shape[1], 1)
losses = []

for iteration in np.arange(0, args["iterations"]):
    iteration_loss = []

    for (batchX, batchY) in next_batch(trainX, trainY, args["batch_size"]):
        predictions = sigmoid_activation(batchX.dot(W))
        error = predictions - batchY
        iteration_loss.append(np.sum(error ** 2))

        d = error * sigmoid_derivative(batchX.dot(W))
        gradient = batchX.T.dot(d)

        W = W - args["alpha"] * gradient
    
    loss = np.average(iteration_loss)
    losses.append(loss)

    if iteration == 0 or (iteration + 1) % 5 == 0:
        print(f"iteration={iteration + 1}, loss={loss}")

test_case_predictions = predict(testX, W)
print(classification_report(testY, test_case_predictions))

plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY[:, 0], s=30)

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["iterations"]), losses)
plt.title("training loss")
plt.xlabel("# of iterations")
plt.ylabel("loss")
plt.show()