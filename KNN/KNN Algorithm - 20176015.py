from scipy.io import arff
import pandas as pd

traindata = arff.loadarff('trainProdSelection.arff')
traindata = pd.DataFrame(data[0])

testdata = arff.loadarff('testProdSelection.arff')
testdata = pd.DataFrame(data[0])

traindata.Type = traindata.Type.str.decode("UTF-8")
traindata.LifeStyle = traindata.LifeStyle.str.decode("UTF-8")
traindata.label = traindata.label.str.decode("UTF-8")

testdata.Type = testdata.Type.str.decode("UTF-8")
testdata.LifeStyle = testdata.LifeStyle.str.decode("UTF-8")
testdata.label = testdata.label.str.decode("UTF-8")

min1 = traindata.Vacation.min()
max1 = traindata.Vacation.max()
traindata.Vacation = traindata.Vacation.apply(lambda x:(x-min1)/(max1-min1))

min1 = traindata.eCredit.min()
max1 = traindata.eCredit.max()
traindata.eCredit = traindata.eCredit.apply(lambda x:(x-min1)/(max1-min1))

min1 = traindata.salary.min()
max1 = traindata.salary.max()
traindata.salary = traindata.salary.apply(lambda x:(x-min1)/(max1-min1))

min1 = traindata.property.min()
max1 = traindata.property.max()
traindata.property = traindata.property.apply(lambda x:(x-min1)/(max1-min1))

min1 = testdata.Vacation.min()
max1 = testdata.Vacation.max()
testdata.Vacation = testdata.Vacation.apply(lambda x:(x-min1)/(max1-min1))

min1 = testdata.eCredit.min()
max1 = testdata.eCredit.max()
testdata.eCredit = testdata.eCredit.apply(lambda x:(x-min1)/(max1-min1))

min1 = testdata.salary.min()
max1 = testdata.salary.max()
testdata.salary = testdata.salary.apply(lambda x:(x-min1)/(max1-min1))

min1 = testdata.property.min()
max1 = testdata.property.max()
testdata.property = testdata.property.apply(lambda x:(x-min1)/(max1-min1))

import math

def euclideanDistance(instance1,instance2,length):
    distance = 0
    if(instance1[0]!=instance2[0]):
        distance += 1
    if(instance1[1]!=instance2[1]):
        distance += 1
    for x in range(2,length,1):        
            distance += pow((instance1[x]-instance2[x]),2)
    return math.sqrt(distance)

import operator 
def getNeighbors(trainingSet, testInstance, k):   
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:            
            correct += 1
    return (correct/float(len(testSet))) * 100.0

def predict(k):
    predictions=[]
    for x in range(len(testdata.values.tolist())):
        neighbors = getNeighbors(traindata.values.tolist(), testdata.values.tolist()[x], k)    
        result = getResponse(neighbors)
        predictions.append(result)
    accuracy = getAccuracy(testdata.values.tolist(), predictions)
    print(k,'--accuracy: ',repr(accuracy))

predict(3)
predict(5)
predict(7)
predict(9)
predict(11)
predict(13)
predict(15)
predict(17)
predict(19)