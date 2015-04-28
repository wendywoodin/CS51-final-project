import numpy as np
import matplotlib.pyplot as plt

class Visualizer:

    def kmeans_knn_k(self, knn_accuracy, ks, kmeans_accuracy):
        plt.plot(ks, knn_accuracy, 'bo', [0], kmeans_accuracy, 'rs')
        plt.xlabel('knn = blue, kmeans = red, values for k on bottom')
        plt.ylabel('accuracy (% correct)')
        plt.title('Accuracy of KNN (with diff. ks) vs Kmeans')
        plt.show()
    
    def knn_k(self, knn_accuracy, ks):
        plt.plot(ks, knn_accuracy, 'bo')
        plt.xlabel('k-values')
        plt.ylabel('accuracy (% correct)')
        plt.title('Accuracy of Ks')
        plt.show()
    
    def kmeans_knn(self, knn_accuracy, kmeans_accuracy):
        plt.plot([0], knn_accuracy, 'bo', [0], kmeans_accuracy, 'rs')
        plt.xlabel('knn = blue, kmeans = red')
        plt.ylabel('accuracy (% correct)')
        plt.title('Accuracy of KNN vs Kmeans')
        plt.show()
    
    def kmeans(self, kmeans_accuracy):
        plt.plot([0], kmeans_accuracy, 'bo')
        plt.ylabel('accuracy (% correct)')
        plt.title('Accuracy of Kmeans')
        plt.show()
    
    def knn(self, knn_accuracy):
        plt.plot([0], knn_accuracy, 'bo')
        plt.ylabel('accuracy (% correct)')
        plt.title('Accuracy of KNN (k-value default = 10)')
        plt.show()
