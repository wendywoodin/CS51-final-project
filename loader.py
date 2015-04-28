import os, struct
from array import array as pyarray
import numpy as np
import warnings

class Loader:

    def __init__(self, dataset, path='.'):
        self.path = path
        self.dataset = dataset

    def load_alldata(self, fname_images, fname_labels, digits):
        # Load in the data into images and labels that we can then test on

        # Check the type of the data
        # Open and read the data
        label_file = open(fname_labels, 'rb')
        magic_number, size = struct.unpack(">II", label_file.read(8))
        if magic_number != 2049:
            warnings.warn("Wrong magic number")
        label_raw = pyarray("b", label_file.read())
        label_file.close()

        image_file = open(fname_images, 'rb')
        magic_number, size, rows, cols = struct.unpack(">IIII", image_file.read(16))
        if magic_number != 2051:
            warnings.warn("Wrong magic number")
        image_raw = pyarray("B", image_file.read())
        image_file.close()
    
        # Initialize arrays in which we can store the values, set all to zeros
        images = np.zeros((size, rows, cols), dtype = np.uint8)
        labels = np.zeros((size, 1), dtype = np.uint8)
    
        # Make an index we can iterate across to load in the values
        index = [k for k in range(size) if label_raw[k] in digits]
        length = len(index)
    
        # Make the data fit nicely into our matrix-like arrays
        for i in range(length):
            images[i] = np.array(image_raw[(index[i]*rows*cols) : ((index[i]+1)*rows*cols)]).reshape((rows,cols))
            labels[i] = label_raw[index[i]]
        
        return images, labels

    
    def load_testing(self, digits):
        fname_images = os.path.join(self.path, 'test_images')
        fname_labels = os.path.join(self.path, 'test_labels')
        
        return self.load_alldata(fname_images, fname_labels, digits)
    
    
    def load_training(self, digits):      
        fname_images = os.path.join(self.path, 'train_images')
        fname_labels = os.path.join(self.path, 'train_labels')
        
        return self.load_alldata(fname_images, fname_labels, digits)
        
    def load_dataset(self, digits=np.arange(10)):
        if self.dataset == "testing":
            return self.load_testing(digits)
        elif self.dataset == "training":
            return self.load_training(digits)
        else:
            raise ValueError ("Invalid dataset type")
        
