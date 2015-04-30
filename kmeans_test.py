import numpy as np
import random
import sys

class KMeanClassification(object):
    def __init__(self, images, labels, k = 10):
        # get data
        size = len(images)

        rows = 28
        cols = 28

        data = np.zeros((size,rows*cols))

        for i in range(size):
            data[i] = images[i].flatten()

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

        import pdb; pdb.set_trace()

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
