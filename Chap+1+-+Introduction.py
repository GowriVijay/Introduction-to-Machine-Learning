
# coding: utf-8

# In[1]:

l = [1,2,4]


# In[3]:

help(l.count)


# In[7]:

import numpy as np
x = np.array([[1,2,3,4]])
print("The numpy array is: {}".format(x))


# In[8]:

x.shape


# In[4]:

from sklearn.datasets import load_iris
iris_dataset = load_iris()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state = 0)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1) 
knn.fit(X_train,y_train)
import numpy as np
x = np.array([[5,2.9,1,0.2]])
print("The numpy array is: {}".format(x))
prediction = knn.predict(X_test)
print("Predicted value for x is : {}".format(prediction))
print("Predicted species for x vlue is : {}".format(iris_dataset['target_names'][prediction]))


# In[5]:

prediction==y_test
import numpy as np
x = np.array([[1,2,3,4]])
print("The numpy array is: {}".format(x))


# In[6]:

np.mean(prediction ==y_test)


# In[7]:

knn.score(prediction,y_test)


# In[8]:

knn.score(X_test, y_test)


# np.mean(prediction ==y_test)
