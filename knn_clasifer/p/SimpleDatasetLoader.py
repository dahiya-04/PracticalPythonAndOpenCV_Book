import numpy as np
import cv2
import os

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        # store the image preprocessor
        self.preprocessors = preprocessors

        # if preprocessors is None, initialize an empty list
        if self.preprocessors is None:
            self.preprocessors = []
    
    def load(self, imagePaths, verbose=-1):
        data = []
        labels = []

        # loop over input images
        for (i, imagePath) in enumerate(imagePaths):
            # load the image and extract class label
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            # apply preprocessors (if any)
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)

            # store processed image and label
            data.append(image)
            labels.append(label)

            # show progress
            if verbose > 0 and (i + 1) % verbose == 0:
                print("INFO processed {}/{}".format(i + 1, len(imagePaths)))

        return (np.array(data), np.array(labels))
