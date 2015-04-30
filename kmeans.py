import numpy as np
import random
import sys
from operator import itemgetter
from collections import defaultdict
from collections import Counter
from scipy.stats import mode

class KMeanClassification(object):
# KMean will generate labelled clusters 
    def __init__(self, images, labels, k = 50):
        # initializes the classification method for kmeans
        # needs itself, a dataset, and a k
        for i in range(len(images)):
            self.dataset.append((images[i],labels[i]))
        to_test = []
        for i in range(100):
            to_test.append(random.randint(0,len(images)))
        dataset2 = []
        for i in to_test:
            dataset2.append(dataset[i])
        dataset1 = []
        for i in range(len(images)):
            if not(i in to_test):
                dataset1.append(dataset[i])
        self.k = k
        self.images = images
        self.labels = labels
        centroid, clusters = self.find_centroid(dataset1)

        
    def euclidian_distance (image1, image2):
    # Retrieve the Euclidian distance 
        return np.linalg.norm(image1 - image2)
        # return np.sqrt(sum((pixel1 - pixel2) ** 2))
    
    def label_assignment (self, dataset, centroid):
    # Give each point a label by finding nearest centroid
    # and assign point to nearest centroid and making it part of 
    # a cluster
        clusters = []
        # Will a touple work for this?
        for x in dataset:
          for c in centroid:
            # I'm not 100% sure that this works becasue of centroid [c[0]]
            closest_centroid = min([self.euclidean_distance(dataset[x][0],centroid[c[0]])], key = lambda val: val[0])
            clusters[closest_centroid].append(x)
        return clusters
    
    def change_centroid(centroid, clusters):
    # Change centroids according to the mean of the points
    # in the clusters
        new_centroid = []
        cluster_images = []
        cluster_labels = []
        for c in clusters:
            cluster_images.append(c[0])
            cluster_labels.append(c[1])
        centroid_image = np.mean(cluster_images, axis = 0)
        return centroid_image
   # This doesn't work?     
    def convergence(centroid, old_centroid):
        return (str(old_centroid) == str(centroid))
    # Returns true if the centroids can no longer be re-defined
        #return (set([tuple(a) for a in old_centroid]) == set([tuple(a) for a in centroid])
        #return collections.Counter(old_centroid) == collections.Counter(centroid))

       # return str(old_centroid) == str(centroid))
    def find_centroid(self, dataset): 
    # Updates centroids
    # old_centroid = random.sample(self.dataset, k)
    # centroid = random.sample(self.dataset, k)
         old_centroid = random.sample(dataset, self.k)
         centroid = random.sample(dataset, self.k)
         while not self.convergence(centroid, old_centroid):
             old_centroid = centroid
             clusters = self.label_assignment(dataset, centroid)

                centroid = self.change_centroid(old_centroid, clusters)    
         return centroid, clusters

   def majority(self, dataset, clusters):
        cluster_labels = []
        centroids = 
        if (self.convergence(centroid, old_centroid) == True):

            for item in clusters:
                x, y = item
                cluster_labels.append(y)

            most_common = mode(cluster_labels)

            #for item in clusters:
                #majority = max((Counter(x).most_common(1)[0]), key=itemgetter(1))[0]

            for index, img in enumerate(centroid):
                # temp = list(centroid[index])
                # temp[0] = most_common
                # centroid[c] = tuple(temp)

                # FOR EACH POINT IN CLUSTER
                    # If centroidpoint and cluster point within range

                        

                        self.dataset[img][] = most_common
                # Does this actually work?
                # new_centroid_label[]
                # centroid = new_centroid_label.items
                # centroid = self.dataset [images][label!!! change this]
    return centroid


   def predictions(self):
    # Split the dataset in half?? How do I do this, anyway, say dataset2 
    # is the real dataset with
    for x in dataset2:
        for c in centroid:
            # I'm not 100% sure that this works becasue of centroid [c[0]]
            closest_centroid = min([self.euclidean_distance(datset2[x][0],centroid[c[0]])], key = lambda val: val[0])
            for y in enumerate(dataset2):
                    temp = list(dataset[y])
                    temp[1] = majority(self, dataset2, clusters)
                    centroid[c] = tuple(temp)
    return dataset2

