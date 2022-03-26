from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2

import plotly as py
import plotly.graph_objs as go
import plotly.express as px


#70000   10

DATA_DIR = "C:\\Users\\Huseyin\\Desktop\\Image_Proccesing\\image_clustering\\dataset2\\train-images-idx3-ubyte"
DATA_LABEL_DIR = "C:\\Users\\Huseyin\\Desktop\\Image_Proccesing\\image_clustering\\dataset2\\train-labels-idx1-ubyte"
DATA_RAW = "C:\\Users\\Huseyin\\Desktop\\Image_Proccesing\\image_clustering\\dataset2\\fashion-mnist_train.csv"
DATA_TEST_RAW = "C:\\Users\\Huseyin\\Desktop\\Image_Proccesing\\image_clustering\\dataset2\\fashion-mnist_test.csv"
WORK_DIR = "C:\\Users\\Huseyin\\Desktop\\Image_Proccesing\\image_clustering\\dataset2"

train = pd.read_csv(DATA_RAW)
test = pd.read_csv(DATA_TEST_RAW)

data = pd.concat([test,train],axis=0,ignore_index=True)
# print(data.head())
# data.info()

X_train = data.drop("label",axis=1)
Y_train = data["label"]

# img = X_train.iloc[69999].values
# img = img.reshape(28,28)
# img = img.astype(np.uint8)
# cv2.imshow("aa",img)
#
# plt.imshow(img,cmap="gray")
# plt.axis("off")

processed_data = np.array(X_train.iloc[:].values)
processed_data = processed_data.astype(np.uint8)

data_count = processed_data[:50000]  #işleme girecek data sayısı 0-70000

pca = PCA(0.90)
pca.fit(data_count)
data_count = pca.transform(data_count)
print("PCA procces",data_count.shape, data_count.size)


k_means = KMeans(init ="k-means++", n_clusters = 10, n_init = 35)
k_means.fit(data_count)
G = len(np.unique(k_means.labels_))
cluster_index= [[] for i in range(G)]
for i, label in enumerate(k_means.labels_,0):
    for n in range(G):
        if label == n:
            cluster_index[n].append(i)
        else:
            continue

clust = 0
print(cluster_index[clust])
for i in range(5):
    img = X_train.iloc[cluster_index[clust][i]].values
    img = img.reshape(28, 28).astype(np.uint8)
    cv2.imshow("1- "+str(i),cv2.resize(img,(280,280)))



print(data_count.shape)
plt.figure(figsize=(10,10))
num = 100
for i in range(1,num):
    plt.subplot(10, 10, i)
    plt.imshow(processed_data[cluster_index[clust][i]].reshape(28, 28), cmap = plt.cm.binary)




layout = go.Layout(
    title='<b>Cluster Visualisation</b>',
    yaxis=dict(
        title='<i>Y</i>'
    ),
    xaxis=dict(
        title='<i>X</i>'
    )
)
colors = ['red', 'green', 'blue', 'purple', 'magenta', 'yellow', 'cyan', 'maroon', 'teal', 'black']
trace = [go.Scatter3d() for _ in range(11)]
for i in range(0, 10):
    my_members = (k_means.labels_ == i)
    index = [h for h, g in enumerate(my_members) if g]
    trace[i] = go.Scatter3d(
        x=data_count[my_members, 0],
        y=data_count[my_members, 1],
        z=data_count[my_members, 2],
        mode='markers',
        marker=dict(size=2, color=colors[i]),
        hovertext=index,
        name='Cluster' + str(i),

    )
fig = go.Figure(
    data=[trace[0], trace[1], trace[2], trace[3], trace[4], trace[5], trace[6], trace[7], trace[8], trace[9]],
    layout=layout)

py.offline.iplot(fig)



plt.show()
cv2.waitKey(0)