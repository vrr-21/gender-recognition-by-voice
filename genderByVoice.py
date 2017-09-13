import pandas as pd
import numpy as np

dataFrame=pd.read_csv('voice.csv')

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(dataFrame[['meanfreq','sd','centroid','meanfun','IQR','median']],dataFrame['label'],random_state=0)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

print("k-NN model trained")
print("Test score:{:.4f}".format((knn.score(X_test,y_test))*100))
