import numpy as np
import random
import sys
from scipy.stats import mode

class KMeanClassification(object):
    def __init__(self, images, labels, k = 5):
        # get data
        # we want to randomly choose 50 points that we are testing
        random.seed()
        # set up a list of 50 randomly chosen indices
        to_test_index = []
        for i in range(len(images)/2):
            to_test_index.append(random.randint(0,len(images)-1))
        # get our answers from the proper indices of the labels

        to_test = []
        for i in range(len(to_test_index)):
            to_test.append(images[to_test_index[i]])

        answers = []
        for i in range(len(to_test_index)):
            answers.append(labels[to_test_index[i]])

        to_train = []
        for i in range(len(images)):
            if not(i in to_test_index):
                to_train.append(images[i])

        size = len(to_test)

        rows = 28
        cols = 28

        data = np.zeros((size,rows*cols))

        for i in range(size):
            data[i] = to_test[i].flatten()

        # define random start centroids
        centroids = self.init_randoms(k, rows*cols)
        clusters = []

        self.i = 0

        self.max_i = 200

        self.last_error = 0

        error = -1

        print("Showing errors for kmeans (watch it converge!)")

        # iterate until convergence
        while not self.convergence(error):
            # assign things to centroids
            clusters = self.update_cluster(data, centroids)
            # update centroids
            centroids = self.update_centroids(clusters, rows*cols)

            error = self.error(centroids, clusters)
            print error

        # Getting the original indices of the datapoints in the clusters

        small_clusters = []

        for i in range(len(clusters)):
            smallest = min(len(clusters[i]), 100)
            small_clusters.append(clusters[i][0:smallest])

        cluster_indices = [[] for i in range(len(centroids))]

        for i in range(len(small_clusters)):
            cluster_indices[i] = [np.where(data == small_clusters[i][x])[0][0] for x in range(len(small_clusters[i]))]

        # Getting the labels of the datapoints by index

        cluster_labels = [[] for i in range(len(cluster_indices))]

        for i in range(len(cluster_indices)):
            cluster_labels[i] = [labels[cluster_indices[i][x]] for x in range(len(cluster_indices[i]))]

        # Assigning labels to centroids

        centroid_labels = []

        for i in range(len(cluster_labels)):
            label_all = mode(cluster_labels[i])
            label = label_all[0]
            centroid_labels.append(label)

        print("{}".format(centroid_labels))


    # return True if convergence
    def convergence(self, error):
        if self.last_error == error or self.i > self.max_i:
            return True
        self.last_error = error
        self.i += 1
        return False

    def error(self, centroids, clusters):
        error = 0
        for i, cluster in enumerate(clusters):
            cluster_center = centroids[i]
            for datum in cluster:
                error += self.distance(datum, cluster_center) ** 2

        return error


    def distance(self,a,b):
        return np.linalg.norm(a-b)

    # returns array of centroids
    def init_randoms(self, k, dimensions):
        return [self.init_random(dimensions) for i in range(k)]

    # returns array of centroids
    def init_random(self, dimensions):
        return [random.randint(0, 255) for j in range(dimensions)]

    # return array of clusters, i.e. [[elements of cluster 1...], [elemeents of cluster 2...], ...]
    def update_cluster(self,data, centroids):
        clusters = [[] for i in range(len(centroids))]
        for datum in data:
            min_idx = np.argmin(np.array([self.distance(datum, centroid) for centroid in centroids]))
            clusters[min_idx].append(datum)
        return clusters

    # returns array of centroids
    def update_centroids(self,clusters, dimensions):
        return [np.mean(x, axis=0) if len(x) > 0 else self.init_random(dimensions) for x in clusters]
