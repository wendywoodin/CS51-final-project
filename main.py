import sys
import re
import argparse

from knn import KNNClassification
#from kmeans import Kmeans
from kmeans_test import KMeanClassification
from loader import Loader
from visualization import Visualizer


class PredictionRunner:

    def run_prediction(self):

        knn_accuracy = []
        kmeans_accuracy = []
        kmeans_k =  self.default_kmeansk
        knn_k = self.default_knnk

        if self.choices[0][0] != "none":
            if self.choices[0][1] == []:
                KMeanClassification(images,labels)
                print("I ran it")
            else:
                for k in self.choices[0][1]:
                    kmeans_accuracy.append(0.0)
                kmeans_k = self.choices[0][1]

        if self.choices[1][0] != "none":
            if self.choices[1][1] == []:
                knn = KNNClassification(images, labels)
                predictions, answers = knn.predictions()
                knn_accuracy = [self.evaluate(predictions, answers)]
            else:
                for k in self.choices[1][1]:
                    knn = KNNClassification(images, labels, k)
                    predictions, answers = knn.predictions()
                    result = self.evaluate(predictions, answers)
                    knn_accuracy.append(result)
                knn_k = self.choices[1][1]

        self.visualize(knn_accuracy, knn_k, kmeans_accuracy, kmeans_k)

    def visualize(self, knn_accuracy, knn_k, kmeans_accuracy, kmeans_k):
        visualize = Visualizer()
        if knn_accuracy != [] and kmeans_accuracy != []:
            visualize.kmeans_knn(knn_accuracy, knn_k, kmeans_accuracy, kmeans_k)
        elif kmeans_accuracy != []:
            visualize.kmeans(kmeans_accuracy, kmeans_k)
        elif knn_accuracy != []:
            visualize.knn(knn_accuracy, knn_k)

    def evaluate(self, predictions, answers):
        index = range(len(predictions))
        correct = 0
        for i in index:
            if predictions[i] == answers[i]:
                correct += 1
        total = len(predictions)
        return (float(correct)/float(total)*100.)



    def __init__(self, args, images, labels):

        self.default_kmeansk = [50]
        self.default_knnk = [10]
        self.choices = [("none", []), ("none", [])]

        if args.kmeans:
            if args.kmeansk:
                self.choices[0] = ("kmeans", args.kmeansk)
            else:
                self.choices[0] = ("kmeans", [])

        if args.knn:
            if args.knnk:
                self.choices[1] = ("knn", args.knnk)
            else:
                self.choices[1] = ("knn", [])

        self.run_prediction()

        sys.exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose the algorithms to test on handwritten digit recognition.")
    parser.add_argument("-m", "--kmeans", help="Run kmeans", action = "store_true")
    parser.add_argument("-n","--knn", help="Run knn, if no k-value specified, default is used", action = "store_true")
    parser.add_argument("-nk", "--knnk", metavar='N', type=int, nargs='+',help= "Choose k-values for knn (type -nk # # #)")
    parser.add_argument("-mk", "--kmeansk", metavar='N', type=int, choices = [10,15,20],help= "Choose k-values for kmeans (type -mk # # #) (choose from 10,15,20)")
    args = parser.parse_args()
    if not(args.kmeans or args.knn):
        parser.print_help()
    else:
        load = Loader("testing")
        images, labels = load.load_dataset()
        PredictionRunner(args, images, labels)
