# Usage: python classification_with_regularization --dataset ~/Downloads/first-image-classifier/dataset/animals
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing.simple_preprocessor import SimplePreprocessor
from pyimagesearch.datasets.simple_dataset_loader import SimpleDatasetLoader
from imutils import paths
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

image_paths = list(paths.list_images(args["dataset"]))

sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(image_paths, verbose=500)
print(data.shape)
data = data.reshape((data.shape[0], data.shape[1] * data.shape[2] * data.shape[3]))

le = LabelEncoder()
labels = le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# The idea:
# the regularization term discourages "overfitting" by introducing an extra loss term that is dependent on the "size" of the weights
# Loss = sum(error) + regularization
# L1 regularization takes the abosolute value of each element in the weights and sums them;
# L2 regularization takes the square of each element in the weights and sums them
# The penalty term (None, L1, L2), or regularization term, is a hyperparameter that can be tuned for optimal results
for r in (None, "l1", "l2"):
    # inputs:
    # loss: the type of loss function
    # penalty: The type of regularization 
    # max_iter: the number of iterations for gradient descent
    # learning_rate: the type of learning rate (alpha term)
    # tol: training error tolerance
    # eta: the alpha term (learning rate)
    # random_state: the random seed. Set to reproduce results
    model = SGDClassifier(loss="log_loss", penalty=r, max_iter=10, learning_rate="constant", tol=1e-3, eta0=0.01, random_state=12)
    model.fit(trainX, trainY)

    acc = model.score(testX, testY)
    print("[INFO] {} penalty accuracy: {:.2f}%".format(r, acc * 100))