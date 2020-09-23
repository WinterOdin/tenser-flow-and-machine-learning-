import tensorflow
import keras
import pandas
import numpy
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


data = pandas.read_csv('student-mat.csv', sep =";")
dataFrame = data[["G1","G2","G3","studytime"]]

predict = "G3"

x = numpy.array(dataFrame.drop([predict], 1))
y = numpy.array(dataFrame[predict])
xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(x,y, test_size=0.1)
"""
#spliting in arrays
best = 0
for j in range(20):
    xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(x,y, test_size=0.1)

    # traning model
    
    linear = linear_model.LinearRegression()
    linear.fit(xTrain,yTrain)
    accuracy = linear.score(xTest,yTest)
    print(accuracy)
    
    if accuracy > best:
        best = accuracy
        with open('studentmodel.pickle','wb') as pickleFile:
            pickle.dump(linear, pickleFile)
"""
pickleIn = open('studentmodel.pickle','rb')
linear   = pickle.load(pickleIn)


#trying to predict the last grade
predictions = linear.predict(xTest)
for grade in range(len(predictions)):
    print(round(predictions[grade]), yTest[grade])

p = "G1"

style.use("ggplot")
pyplot.scatter(dataFrame[p],dataFrame["G3"])
pyplot.xlabel(p)
pyplot.ylabel("final grade")
pyplot.show()