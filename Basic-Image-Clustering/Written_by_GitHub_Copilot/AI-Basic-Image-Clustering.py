import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


"""pca algorithm"""  #written by copilot-AI
def PCA(X, ndim):
    """
    X is a numpy.ndarray of shape (n, d) where:
    n is the number of data points
    d is the number of dimensions in each point
    ndim is the new dimensionality of the transformed X
    Returns: T, a numpy.ndarray of shape (n, ndim) containing
    the transformed version of X
    """
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    X_cov = np.matmul(X_centered.T, X_centered) / (X.shape[0] - 1)
    eig_vals, eig_vecs = np.linalg.eig(X_cov)
    idx = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]
    T = np.matmul(X_centered, eig_vecs[:, :ndim])
    return T

"""kmeans algorithm"""  #written by copilot-AI
def kmeans(X, k, max_iter=100,plot_progress=True):
    """
    X is a numpy.ndarray of shape (n, d) containing the dataset
    n is the number of data points
    d is the number of dimensions for each point
    k is the number of clusters
    max_iter is the maximum number of updates before the algorithm terminates
    plot_progress is a boolean that indicates whether or not to plot progress
    Returns: C, the centroid locations of the clusters
    """
    n, d = X.shape
    C = np.random.rand(k, d)
    C = X[np.random.choice(n, k, replace=False)]
    prev_assignments = None
    for i in range(max_iter):
        # Assign each point to the closest centroid
        assignments = np.argmin(np.linalg.norm(X[:, None] - C[None], axis=2), axis=1)
        # If the assignments are the same as the previous iteration, this is complete
        if prev_assignments is not None and np.array_equal(assignments, prev_assignments):
            break
        # Update centroids to be the mean of the points assigned to them
        for j in range(k):
            C[j] = X[assignments == j].mean(axis=0)
        if plot_progress:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=assignments, s=10, alpha=0.5)
            ax.scatter(C[:, 0], C[:, 1], C[:, 2], c='black', s=50, alpha=1)
            plt.show()
        prev_assignments = assignments
    return C





DATA_DIR = "C:\\Users\\Huseyin\\Desktop\\Image_Proccesing\\image_clustering\\dataset2\\train-images-idx3-ubyte"
DATA_LABEL_DIR = "C:\\Users\\Huseyin\\Desktop\\Image_Proccesing\\image_clustering\\dataset2\\train-labels-idx1-ubyte"
DATA_RAW = "C:\\Users\\Huseyin\\Desktop\\Image_Proccesing\\image_clustering\\dataset2\\fashion-mnist_train.csv"
DATA_TEST_RAW = "C:\\Users\\Huseyin\\Desktop\\Image_Proccesing\\image_clustering\\dataset2\\fashion-mnist_test.csv"
WORK_DIR = "C:\\Users\\Huseyin\\Desktop\\Image_Proccesing\\image_clustering\\dataset2"

train = pd.read_csv(DATA_RAW)
test = pd.read_csv(DATA_TEST_RAW)

data = pd.concat([test,train],axis=0,ignore_index=True)
X_train = data.drop("label",axis=1)
Y_train = data["label"]

processed_data = np.array(X_train.iloc[:].values)
processed_data = processed_data.astype(np.uint8)

num = 10000   # <------------------- enter the number of data to process
data_count = processed_data[:num]
print("before pca",data_count.shape)

data_count =PCA(data_count, 10)
data_count = np.around(data_count, 2)
print("after pca",data_count.shape)

cc = kmeans(data_count,10,plot_progress=False)
cc = np.around(cc,2)


"""plot data and centroids"""  #written by copilot-AI
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data_count[:num, 0], data_count[:num, 1], data_count[:num, 2], c=Y_train[:num], s=10, alpha=0.5)
ax.scatter(cc[:, 0], cc[:, 1], cc[:, 2], c='black', s=50, alpha=1)



"""match data with classes"""  #written by copilot-AI
assignments = np.argmin(np.linalg.norm(data_count[:, None] - cc[None], axis=2), axis=1)
#print(assignments)



"""Allocate indexes of class 1"""  #written by copilot-AI
Selected_class = 0 # <------------------- select class
indexes = []
for i in range(len(assignments)):
    if assignments[i] == Selected_class:
        indexes.append(i)



plt.figure(figsize=(10,10))
num = 100
for i in range(1,num):
    plt.subplot(10, 10, i)
    plt.imshow(processed_data[indexes[i]].reshape(28, 28), cmap = plt.cm.binary)
plt.show()




