import numpy as np
import cv2
import os

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors

        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        # initialize list of features and labels
        data = []
        labels = []

        # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
            # assume that the path to images have the following format:
            # /path/to/dataset/{class}/{image name}.jpg
            image = cv2.imread(imagePath)
            class_label = imagePath.split("/")[-2]

            if self.preprocessors is not None:
                # loop over the preprocessors and apply each to the image
                for preprocessor in self.preprocessors:
                    image = preprocessor.preprocess(image)

                # load the processed image into the `data` list, 
                # treating it as a "feature vector"
                data.append(image)
                # save the corresponding label of this image
                labels.append(class_label)

                if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                    print(f"processed {i + 1} / {len(imagePaths)}")

        # return a tuple of the data and labels
        return (np.array(data), np.array(labels))