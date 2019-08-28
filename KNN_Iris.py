#read iris data 
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np

iris = datasets.load_iris()
#X features and y label 
X = iris.data
y = iris.target

#split data 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=4)
#model 
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test,y_pred))

#predict
x_new = np.array([[4,3,3,4]])
#print("X_new.shape: {}".format(x_new.shape))

# make a predition based on the above sample
prediction = knn.predict(x_new)
print("Prediction:",prediction)
print("Predicted target name:{}".format(iris['target_names'][prediction]))
