import sys
import re
import argparse

from knn import KNNClassification
#from kmeans import Kmeans
from loader import Loader
from visualization import Visualizer


class PredictionRunner:

    def run_prediction(self):
       num_inputs = len(self.prediction_algorithms)
       visualize = Visualizer()
       
       if num_inputs == 3:
           print("running both kmeans and knn with k")
           knn_accuracy = []
           for k in self.prediction_algorithms[2]:
               knn = KNNClassification(images, labels, k)
               predictions, answers = knn.predictions()
               result = self.evaluate(predictions, answers)
               knn_accuracy.append(result)
           kmeans_accuracy = 0.0
           visualize.kmeans_knn_k(knn_accuracy, self.prediction_algorithms[2], kmeans_accuracy)
           for predict in knn_accuracy:
               print("{}".format(predict))
       
       elif num_inputs == 2:
           if self.prediction_algorithms[0] == "knn":
               print("running knn with k")
               knn_accuracy = []
               for k in self.prediction_algorithms[1]:
                   knn = KNNClassification(images, labels, k)
                   predictions, answers = knn.predictions()
                   result = self.evaluate(predictions, answers)
                   knn_accuracy.append(result)
               visualize.knn_k(knn_accuracy, self.prediction_algorithms[1])
               for predict in knn_accuracy:
                   print("{}".format(predict))
           else:
               print("running knn with no k and kmeans")
               knn = KNNClassification(images, labels)
               predictions, answers = knn.predictions()
               knn_accuracy = self.evaluate(predictions, answers)
               kmeans_accuracy = 0.0
               visualize.kmeans_knn(knn_accuracy, kmeans_accuracy)
               print("{}".format(knn_accuracy))
               
       elif num_inputs == 1:
           if self.prediction_algorithms[0] == "kmeans":
               print("running kmeans")
               kmeans_accuracy = 0.0
               visualize.kmeans(kmeans_accuracy)
           else:
               print("running knn")
               knn = KNNClassification(images, labels)
               predictions, answers = knn.predictions()
               knn_accuracy = self.evaluate(predictions, answers)
               visualize.knn(knn_accuracy)
               print("{}".format(knn_accuracy))
       else:
           raise ValueError ("Invalid inputs")
    
    def evaluate(self, predictions, answers):
        index = range(len(predictions))
        correct = 0
        for i in index:
            if predictions[i] == answers[i]:
                correct += 1
        total = len(predictions)
        return (float(correct)/float(total)*100.)

        
        
    def __init__(self, args, images, labels):

        self.prediction_algorithms = []   

        if args.kmeans:
            self.prediction_algorithms.append("kmeans")
        
        if args.knn:
            self.prediction_algorithms.append("knn")
            
        if args.knn and args.kvalues:
            self.prediction_algorithms.append(args.kvalues)
        
        
            
        self.run_prediction()
        
        sys.exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose the algorithms to test on handwritten digit recognition.")
    parser.add_argument("-m", "--kmeans", help="Run kmeans", action = "store_true")
    parser.add_argument("-n","--knn", help="Run knn, if no k-value specified, default is used", action = "store_true")
    parser.add_argument("-k", "--kvalues", metavar='N', type=int, nargs='+',help= "Choose k-values for knn (type -k # # #)")
    args = parser.parse_args()
    if not(args.kmeans or args.knn):
        parser.print_help()
    else:
        load = Loader("testing")
        images, labels = load.load_dataset()
        PredictionRunner(args, images, labels)
