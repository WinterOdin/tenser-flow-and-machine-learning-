import tensorflow
import keras
import pandas
import numpy
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pandas.read_csv("car.data")
#converting text to numeric data

pre = preprocessing.LabelEncoder()
buying  = pre.fit_transform(list(data["buying"]))
maint   = pre.fit_transform(list(data["maint"]))
lug     = pre.fit_transform(list(data["lug_boot"]))
safety  = pre.fit_transform(list(data["safety"]))
cls     = pre.fit_transform(list(data["class"]))

predict = 'class'

x = list(zip(buying,maint,lug,safety,cls))
y = list(cls)

xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(x,y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=9)
model.fit(xTrain,yTrain)
predicted = model.predict(xTest)
names = ['unacc','acc','good','vgood']

for x in range(len(predicted)):
    print("predicted: ", names[predicted[x]], "Data: ", xTest[x],"Actual: ", names[yTest[x]])