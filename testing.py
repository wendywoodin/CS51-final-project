import numpy as np

images = np.zeros((5,3,3))

'''
Trying to do PCA conceptually on a smaller set
'''

for i in range(len(images)):
    images[i] = np.array([2*i,2*i+1,2*i+2,2*i+3,2*i+4,\
        2*i+5,2*i+6,2*i+7,2*i+8]).reshape((3,3))

# these are the means of the rows, not the columns, because of the
# way that images are stored as 28 rows of 28 elements
mean_1 = np.mean(images[:][:,0])
mean_2 = np.mean(images[:][:,1])
mean_3 = np.mean(images[:][:,2])

# Get some variables abstracted out
rows = 3
cols = 3

# But! it's not a 3 dimensional sample, it's a 3x3 dimensional sample
# (Or a 28x28 in terms of the actual dataset)
mean_matrix = np.zeros((rows,cols))

for i in range(rows):
    for j in range(cols):
        mean_matrix[i][j] = np.mean(images[:,i,j])


# Moving on to covariance, a tricky one!

# Initialize some sort of array for this
mat_for_cov = []
for i in range(rows):
    mat_for_cov.append(([]))
    for j in range(cols):
        mat_for_cov[i].append(([]))

for i in range(rows):
    for j in range(cols):
        mat_for_cov[i][j] = images[:,i,j]

np.cov([all_samples[0,:],all_samples[1,:],all_samples[2,:]])

'''
Okay so apparently this only works with vectors, not matrices
'''

# We have to first flatten into a 28*28 (784) long vector
size = len(images)
workable = np.zeros((size,rows*cols))

for i in range(size):
    workable[i] = images[i].flatten()

# Now let's get a variable for the length of the new vector thing

length = len(workable[0])

# Then we can calculate the mean *vector* (not a matrix now)

mean_vector = []
for i in range(length):
    mean_vector.append([np.mean(workable[:,i])])


# Covariance matrix can't handle the 10,000 point dataset :(
# Scatter matrix by hand, because memory

scatter_matrix = np.zeros((length,length))
for i in range(size):
    scatter_matrix += (workable[i]-mean_vector).dot((workable[i]\
        -mean_vector).T)


# Seems legit, continuing with eigenvalues

eig_val, eig_vec = np.linalg.eig(scatter_matrix)


# Now we need to list our eigenvalues with our eigenvectors
# So we can sort and choose

eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]

eig_pairs.sort(key = lambda val: val[0])
eig_pairs.reverse()

# Make a matrix with some picked

# But first, define how many to pick
k = 3
to_be = []

# Gotta make a something for hstack to use, and make it dynamically
# Depending on k
for i in range(k):
    to_be.append((eig_pairs[i][1].reshape(length,1)))

# Now we get to actually make the matrix
matrix_pca = np.hstack((to_be))

# Transform the old space with our matrix!
# transformed = matrix_pca.T.dot(workable)
transformed = workable.dot(matrix_pca)

'''
Checking out index finding
'''
import numpy as np
from scipy.stats import mode

rows = 3
cols = 3

images = np.zeros((5,3,3))
for i in range(len(images)):
    images[i] = np.array([2*i,2*i+1,2*i+2,2*i+3,2*i+4,\
        2*i+5,2*i+6,2*i+7,2*i+8]).reshape((3,3))

labels = np.zeros((5,1))
labels[0] = 0
labels[1] = 0
labels[2] = 1
labels[3] = 1
labels[4] = 1

size = len(images)
data = np.zeros((size,rows*cols))

for i in range(size):
    data[i] = images[i].flatten()

centroids = np.array([[1,2,3,4,5,6,7,8,9],[0,1,2,3,4,5,6,7,8]])

clusters = np.array([[data[4],data[2]],[data[0],data[1],data[3]]])

# Getting the original indices of the datapoints in the clusters

cluster_indices = [[] for i in range(len(centroids))]

for i in range(len(clusters)):
    cluster_indices[i] = [np.where(data == clusters[i][x])[0][0] for x in range(len(clusters[i]))]

# Getting the labels of the datapoints by index

cluster_labels = [[] for i in range(len(cluster_indices))]

for i in range(len(cluster_indices)):
    cluster_labels[i] = [labels[cluster_indices[i][x]] for x in range(len(cluster_indices[i]))]

# Assigning labels to centroids

centroid_labels = []

for i in range(len(cluster_labels)):
    centroid_labels.append(mode(cluster_labels[i])[0])

print("{}".format(centroid_labels))
