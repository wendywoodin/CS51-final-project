import numpy as np
import random
import sys
from scipy.stats import mode
import pdb

class KMeanClassification(object):
    def __init__(self, images, labels, k = 10):
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
            to_test.append(images[to_test_index[i]].flatten())

        answers = []
        for i in range(len(to_test_index)):
            answers.append(labels[to_test_index[i]])

        to_train = []
        for i in range(len(images)):
            if not(i in to_test_index):
                to_train.append(images[i].flatten())

        to_train_labels = []
        for i in range(len(images)):
            if not(i in to_test_index):
                to_train_labels.append(labels[i])

        size = len(to_test)

        rows = 28
        cols = 28


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
            clusters = self.update_cluster(to_train, centroids)
            # update centroids
            centroids = self.update_centroids(clusters, rows*cols)

            error = self.error(centroids, clusters)
            print error

        # Getting the original indices of the datapoints in the clusters

        # Get rid of the centroids without points associated
        new_points = []
        for i in range(len(clusters)):
            if len(clusters[i]) != 0:
                new_points.append(i)
        new_clusters = []
        new_centroids = []
        for i in range(len(clusters)):
            if i in new_points:
                new_clusters.append(clusters[i])
                new_centroids.append(centroids[i])

        # Work with some smaller clusters when checking indices (for memory/time)
        # Also turn it into a list for later so we can do index checking
        small_clusters = []

        for i in range(len(new_clusters)):
            smallest = min(len(new_clusters[i]), 100)
            small_clusters.append(new_clusters[i][0:smallest])

        # get the indices of the items in the cluster

        # first we have to do some odd things with lists/arrays
        list_train = []
        for i in range(len(to_train)):
            list_train.append(to_train[i].tolist())

        cluster_indices = [[] for i in range(len(new_centroids))]
        for i in range(len(small_clusters)):
            cluster_indices[i] = [list_train.index(small_clusters[i][x].tolist()) for x in range(len(small_clusters[i]))]

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

        pdb.set_trace()
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
