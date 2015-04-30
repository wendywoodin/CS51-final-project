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
        self.k = k
        self.images = images
        self.labels = labels
        
    def euclidian_distance (image1, image2):
    # Retrieve the Euclidian distance 
        return np.linalg.norm(pixel1 - pixel2)
        # return np.sqrt(sum((pixel1 - pixel2) ** 2))
    
    def label_assignment (self, dataset, centroid):
    # Give each point a label by finding nearest centroid
    # and assign point to nearest centroid and making it part of 
    # a cluster
        clusters = []
        # Will a touple work for this?
        for x in self.dataset:
          for c in centroid:
            # I'm not 100% sure that this works becasue of centroid [c[0]]
            closest_centroid = min([self.euclidean_distance(self.dataset[x][0],centroid[c[0]])], key = lambda val: val[0])
            clusters[closest_centroid].append(x)
        return clusters
    
    def change_centroid(centroid, clusters):
    # Change centroids according to the mean of the points
    # in the clusters
        new_centroid = []
        keys = values.sort(key = lambda val: val[0]) 
        # Do I need to use keys for this problem? 
        for x in keys:
            new_centroid.append(np.mean(clusters[x])) 
        return new_centroid
        
   # This doesn't work?     
    def convergence(centroid, old_centroid):
        return (str(old_centroid) == str(centroid))
    # Returns true if the centroids can no longer be re-defined
        #return (set([tuple(a) for a in old_centroid]) == set([tuple(a) for a in centroid])
        #return collections.Counter(old_centroid) == collections.Counter(centroid))

       # return str(old_centroid) == str(centroid))
    def find_centroid(self, dataset, k = 50): 
    # Updates centroids
    # old_centroid = random.sample(self.dataset, k)
    # centroid = random.sample(self.dataset, k)
         old_centroid = random.sample(self.dataset, k)
         centroid = random.sample(self.dataset, k)
         while not convergence(centroid, old_centroid):
             old_centroid = centroid
             clusters = label_assignment(self.dataset, centroid)
             centroid = change_centroid(old_centroid, clusters)    
         return centroid
         return clusters

   def majority(self, dataset, clusters):
        cluster_labels = []
        if (self.convergence(centroid, old_centroid) == True):

            for item in clusters:
                x, y = item
                cluster_labels.append(y)

            most_common = mode(cluster_labels)






            #for item in clusters:
                #majority = max((Counter(x).most_common(1)[0]), key=itemgetter(1))[0]

             



            majority = 
            for index, item in enumerate(centroid):
                temp = list(centroid[index])
                temp[0] = most_common
                centroid[c] = tuple(temp)
                # Does this actually work?
                # new_centroid_label[]
                # centroid = new_centroid_label.items
                # centroid = self.dataset [images][label!!! change this]
    return centroid


   def predictions(self):
    # Split the dataset in half?? How do I do this, anyway, say dataset2 
    # is the one I split in half
    dataset1
    # make a copy called dataset2
    dataset2 = list(dataset1)
    for x in dataset2:
        for c in centroid:
            # I'm not 100% sure that this works becasue of centroid [c[0]]
            closest_centroid = min([self.euclidean_distance(datset2[x][0],centroid[c[0]])], key = lambda val: val[0])
            for y in enumerate(dataset2):
                    temp = list(dataset[y])
                    temp[1] = majority(self, dataset2, clusters)
                    centroid[c] = tuple(temp)
    return dataset2


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
    # return the predictions and the answers
    return all_preds, answers
