import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle


iris_data = pd.read_csv('iris.data',sep=',',names=['sepal length in cm','sepal width in cm','petal length in cm','petal width in cm','target'])
iris_data.tail()
print(iris_data.isnull().sum(axis = 0))
print(iris_data.isnull().sum(axis = 1))


sns.heatmap(iris_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
iris_data['sepal length in cm'].plot()
iris_data['sepal width in cm'].plot()
sns.countplot(iris_data['target'],data=iris_data)
sns.scatterplot(iris_data['sepal length in cm'],iris_data['sepal width in cm'],hue=iris_data['target'])
sns.scatterplot(iris_data['petal length in cm'],iris_data['petal width in cm'],hue=iris_data['target'])
data = iris_data


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['target'] = le.fit_transform(data['target'])


data.head()

plt.figure(figsize = (10,7))
sns.lineplot(data=data)

from sklearn.model_selection import train_test_split
X = data.iloc[:,[0,1,2,3]].values

X

X = data.iloc[:,[0,1,2,3]].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=4)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_train.shape)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model = pickle.load(open('model.pkl','rb'))
y_pred = model.predict(X_test)
y_pred
cm
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
