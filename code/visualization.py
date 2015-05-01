import numpy as np
import matplotlib.pyplot as plt

class Visualizer:

    def kmeans_knn(self, knn_accuracy, knn_k, kmeans_accuracy, kmeans_k):
        plt.plot(knn_k, knn_accuracy, 'bo', kmeans_k, kmeans_accuracy, 'rs')
        plt.xlabel('knn = blue, kmeans = red, values for k on bottom')
        plt.ylabel('accuracy (% correct)')
        plt.title('Accuracy of KNN (with diff. ks) vs Kmeans')
        plt.show()
    
    def knn(self, knn_accuracy, knn_k):
        plt.plot(knn_k, knn_accuracy, 'bo')
        plt.xlabel('k-values')
        plt.ylabel('accuracy (% correct)')
        plt.title('Accuracy of Ks')
        plt.show()
    
    def kmeans(self, kmeans_accuracy, kmeans_k):
        plt.plot(kmeans_k, kmeans_accuracy, 'rs')
        plt.ylabel('accuracy (% correct)')
        plt.title('Accuracy of Kmeans')
        plt.show()

