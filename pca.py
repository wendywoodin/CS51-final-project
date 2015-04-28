import numpy as np
from loader import Loader

# Note: This code, while it appears to work on a small, 9 member
# 3x3 dataset I made for myself, cannot work on the larger MNIST
# dataset. Places where the code fails in the appliance are marked.
# Using my computer outside the appliance gets us slightly farther.

load = Loader("testing")

images,labels = load.load_dataset()

# Some variables we'll use
size = len(images)
rows = 28
cols = 28
k = 50


# We have to first flatten into a 28*28 (784) long vector
workable = np.zeros((size,rows*cols))

for i in range(size):
    workable[i] = images[i].flatten()    

# Now let's get a variable for the length of the new vector thing
length = len(workable[0])

# Then we can calculate the mean vector
mean_vector = []
for i in range(length):
    mean_vector.append([np.mean(workable[:,i])])


# Covariance matrix can't handle the 10,000 point dataset :(
# Scatter matrix by hand, because memory
# Now we get no error thrown but see below
# (note, cannot calculate scatter matrix either, it hangs)
# (works outside of the apppliance, as does the built in covariance
#  function)
scatter_matrix = np.zeros((length,length))
for i in range(size):
    scatter_matrix += (workable[i]-mean_vector).dot((workable[i]\
        -mean_vector).T)


# Continuing with finding the eigenvalues and eigenvectors
# (note, this step is beyond the ability of not only the appliance
#  but also my computer to calculate)
eig_val, eig_vec = np.linalg.eig(scatter_matrix)


# Now we need to list our eigenvalues with our eigenvectors
# So we can sort and choose those with the largest eigenvalues
# (these are the eigenvectors that account for the most variance
#  in the data)
eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]

eig_pairs.sort(key = lambda val: val[0])
eig_pairs.reverse()

# Make a matrix with some eigenvectors picked, these are determined
# by the "k above"

# Set up "to_be", which are the eigenvalues to be chosen
to_be = []

# Gotta make a something for hstack to use, and make it dynamically
# Depending on k
for i in range(k):
    to_be.append((eig_pairs[i][1].reshape(length,1)))

# Now we get to actually make the matrix
matrix_pca = np.hstack((to_be))

# Transform the old space with our matrix!
transformed = workable.dot(matrix_pca)
