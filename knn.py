import warnings
import numpy as np
import random
import sys

# Note: Looked at code from http://andrew.gibiansky.com/blog/machine-learning/k-nearest-neighbors-simplest-machine-learning/
class KNNClassification(object):
    def __init__(self, images, labels, k = 20):
        # initializes the classification method for knn
        # needs itself, a dataset, and a k
        self.dataset = []
        for i in range(len(images)):
            self.dataset.append((images[i],labels[i]))
        self.k = k
        self.images = images
        self.labels = labels

    def predict (self, point, index):
        # the prediction function, takes in itself and a point to be classified
        # uses the function distance, which takes the distance btwn 2 data pts

        # we make sure to not test the point against itself
        distances = []
        for i in range(len(self.dataset)):
            if i != index:
                distances.append(self.euclidean_distance(self.dataset[i][0], point))
            if i == index:
                distances.append(sys.maxint)

        values = zip(distances, self.dataset)
        values.sort(key = lambda val: val[0])

        prediction = self.majority([value[1][1] for value in values[0:self.k]])
        return prediction

    def euclidean_distance (self,image1, image2):
        # calculates the distance between two images in the database
        # we'll start out with the simplest (euclidean) and then optimize
        return np.linalg.norm(image1-image2)#sum((image1 - image2)**2)

    # Wrong code???
    def majority(self,votes):
        options = [0,0,0,0,0,0,0,0,0,0]
        for vote in votes:
            # print (vote)
            options[vote] += 1
        highest = (sorted(options)).pop()
        if options.count(highest) > 1:
            warnings.warn("Multiple digits tied for best in knn")
        return options.index(highest)


    def predictions(self):
        # we want to randomly choose 50 points that we are testing
        random.seed()
        # set up a list of 50 randomly chosen indices
        to_test = []
        for i in range(50):
            to_test.append(random.randint(0,len(self.images)))
        # set up a list of lists for the return of self.predict
        all_preds = [[] for _ in range(len(to_test))]
        for i in range(len(to_test)):
            all_preds[i] = self.predict(self.images[to_test[i]], to_test[i])
        # get our answers from the proper indices of the labels
        answers = []
        for i in range(len(to_test)):
            answers.append(self.labels[to_test[i]])
            # print(self.labels[to_test[i]])
        # return the predictions and the answers
        return all_preds, answers
