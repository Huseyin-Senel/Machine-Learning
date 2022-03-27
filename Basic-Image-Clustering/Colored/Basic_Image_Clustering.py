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

#5000   20

DATA_DIR = "C:\\Users\\Huseyin\\Desktop\\Image_Proccesing\\image_clustering\\clothing-dataset-master\\images"
DATA_RAW = "C:\\Users\\Huseyin\\Desktop\\Image_Proccesing\\image_clustering\\clothing-dataset-master\\images.csv"
WORK_DIR = "C:\\Users\\Huseyin\\Desktop\\Image_Proccesing\\image_clustering\\clothing-dataset-master"

data = pd.read_csv(DATA_RAW)
print(data.head(10))
#data.info()


size=200
mini_images =np.zeros((size,size,1), np.uint8)
processed_data = np.zeros((1, size*size), np.uint8)

for i in range(1500):
    img_path = os.path.join(DATA_DIR, data["image"][i] + ".jpg")

    img = cv2.imread(img_path)
    if img is None:
        print("image not found -",i)
        continue
    #cv2.imshow("img", img)
    # img1 = cv2.imread(img_path, 0)
    # cv2.imshow("img1", img1)

    m_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    m_mask[int(img.shape[0] / 3):int((img.shape[0] / 3) * 2), int(img.shape[1] / 3):int((img.shape[1] / 3) * 2)] = 1
    renk = cv2.mean(img, m_mask)
    hsv_renk = cv2.cvtColor(np.uint8([[renk]]), cv2.COLOR_BGR2HSV)[0][0]  # medyan rengi hsv ye dönüştü

    tol = 40
    ll = np.array(
        [int(hsv_renk[0] - (tol / 2)), hsv_renk[1] - tol, hsv_renk[2] - tol])  # 360-100-100  ==  180 - 255 - 255
    ul = np.array([int(hsv_renk[0] + (tol / 2)), hsv_renk[1] + tol, hsv_renk[2] + tol])

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, ll, ul)
    res = cv2.bitwise_and(img, img, mask=mask)
    th10 = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    # ret, th10 = cv2.threshold(th10, 10, 255, cv2.THRESH_BINARY)
    th10 = cv2.resize(th10, (size, size))

    mini_images = np.dstack([mini_images,np.array(th10)])

    a = th10.flatten()
    processed_data = np.vstack([processed_data, np.array(a)])

print("image procces",processed_data.shape, processed_data.size)
print("shape",mini_images.shape)
pca = PCA(0.95)
pca.fit(processed_data)

processed_data = pca.transform(processed_data)
print("PCA procces",processed_data.shape, processed_data.size)


k_means = KMeans(init ="k-means++", n_clusters = 20, n_init = 35)
k_means.fit(processed_data)
G = len(np.unique(k_means.labels_))

cluster_index= [[] for i in range(G)]
for i, label in enumerate(k_means.labels_,0):
    for n in range(G):
        if label == n:
            cluster_index[n].append(i)
        else:
            continue


print(cluster_index)
clust = 0                           #Görüntülenecek Class Seçimi
print(cluster_index[clust])

draw = True
print(int(len(cluster_index[clust])/10))
if int(len(cluster_index[clust])/10) == 0:
    print("So few class image. select another class or restart clustering")
    draw = False
elif int(len(cluster_index[clust])/10) > 4:
    row_num=4
else:
    row_num=int(len(cluster_index[clust])/10)

if draw:
    plt.figure(figsize=(10, 10));
    print(row_num)
    for i in range(1, row_num * 10):
        plt.subplot(row_num, 10, i)  # (satır sayısı, her satırdaki sütun sayısı , item sayısı)
        plt.imshow(mini_images[:, :, cluster_index[clust][i]].reshape(size, size), cmap=plt.cm.binary)

# for i in cluster_index[3][0:5]:
#     cv2.imshow("4-" + str(i), cv2.resize(mini_images[:, :, i], (200, 200)))




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
        x=processed_data[my_members, 0],
        y=processed_data[my_members, 1],
        z=processed_data[my_members, 2],
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