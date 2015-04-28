import numpy as np
import random
from collections import defaultdict

class KMeanClassification(object):
# KMean will generate labelled clusters 
    def __init__(self, dataset, k):
    # K-men takes in two inputs, the dataset
        self.dataset = dataset
        self.k = k
        
    def euclidian_distance (pixel1, pixel2):
    # Retrieve the Euclidian distance 
        return np.linalg.norm(pixel1 - pixel2)
        # return np.sqrt(sum((pixel1 - pixel2) ** 2))
    
    def label_assignment (self, dataset, centroid):
    # Give each pixel point a label by finding nearest centroid
    # and assign point to nearest centroid
        clusters = []
        for x in self.dataset
            closest_points = min([self.euclidean_distance(x[0],point) for x in self.dataset], key = lambda val: val[0])
            clusters[closest_points].extend(x)
        return clusters
                
    def initial_centroid (self, dataset, k)
    # Randomly choose k items and make it the initial centroids
    # Adapted from Stanford's CS221 class 
        numFeatures = dataset.getNumFeatures()
        centroid = getRandomCentroids(numFeatures, k)
    
    def evaluate_centroids (self, dataset, k): 
    # Updates centroids
        new_centroid = []
        values.sort(key = lambda val: val[0]) 
        for x in key
            new_centroid.extend(np.mean(clusters))
        return new_centroid
    
    # def update_clusters (self, dataset, centroid)
    # Update clusters
    
    def convergence (new_centroid, centroid) 
    # Returns true if the centrois can no longer be re-defined
        return new_centroid = centroid
    
    def performance_evaluation(predictions, answers):
    # Evaluator will run this
        
