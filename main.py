import json
import requests
import pandas as pd
import numpy as np
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


#Data File read
gene_train = pd.read_table(r"/content/drive/MyDrive/Clustering_competition/data_tr.txt", header=None)
gene_test=pd.read_table(r'/content/drive/MyDrive/Clustering_competition/data_ts.txt', header=None)

#MinMaxScalar :
np.random.seed(0)
scaler = MinMaxScaler(feature_range=(0, 2))
scaler.fit(gene_train)
gene_data2 = scaler.transform(gene_train)
minmax = scaler.transform(gene_test)

#PCA :
pca = PCA(n_components=100, random_state=0)
pca.fit(gene_data2)
gene_data= pca.transform(gene_data2)
y = pca.transform(minmax)

#Birch Model:
print("=============Birch-Clustering================")
birch_model = Birch(branching_factor = 100, n_clusters=16, threshold=2)
birch_model.fit(gene_data)

Birch_Prediction = birch_model.predict(y).tolist()
BSil_score = silhouette_score(gene_data, birch_model.labels_, metric='euclidean')
url = "https://www.csci555competition.online/scoretest"
print('Loading Predictions to API..')
payload_MShift = json.dumps([Birch_Prediction])
headers = {
    'Content-Type': 'application/json'
}
print('API Calling..')
response = requests.request("POST", url, headers=headers, data=payload_MShift)
print('=======Result==========')
print('Birch Accuracy Result: %', response.text)
print('Birch Silhouette Score: %.3f' % BSil_score)

