import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics


cancer = datasets.load_breast_cancer()

x = cancer.data
y = cancer.target

xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(x,y, test_size=0.3)

names = ['malignat', 'begin']

clf = svm.SVC(kernel="linear", C=2)
clf.fit(xTrain,yTrain)

predictOne = clf.predict(xTest)

accuracy = metrics.accuracy_score(yTest, predictOne)

print(accuracy)