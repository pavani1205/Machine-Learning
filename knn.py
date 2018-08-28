from scipy.io import arff
import pandas as pd
import math
import operator

def reading():
    traindata=arff.loadarff('trainProdSelection.arff')
    df=pd.DataFrame(traindata[0])
    df.Type=df.Type.str.decode("utf-8")
    df.LifeStyle=df.LifeStyle.str.decode("utf-8")
    df.label=df.label.str.decode("utf-8")
    # similarityMatrix=pd.read_excel("similaritymatrix.xls")
    # customer_type=similarityMatrix[:5]
    # lifeStyle=similarityMatrix[7:]
    # lifeStyle.columns=["spend<<saving","spend<saving","spend>saving",
    #                    "spend>>saving","None"]
    # lifeStyle.drop(['None'],axis=1)
    traindata=arff.loadarff('trainProdSelection.arff')
    df=pd.DataFrame(traindata[0])
    df.Type=df.Type.str.decode("utf-8")
    df.LifeStyle=df.LifeStyle.str.decode("utf-8")
    df.label=df.label.str.decode("utf-8")
def eucliddistance(instance1,instance2,length):
    distance=0
    for x in range(length):
        distance+=pow((instance1[x]-instance2[x]),2)
    return math.sqrt(distance)

def neighbours(trainingSet,testInstance,k):
    distances=[]
    length=len(testInstance)-1
    for x in range(len(trainingSet)):
        dist=eucliddistance(testInstance,trainingSet[x],length)
        distances.append((trainingSet[x],dist))
    distances.sort(key=operator.itemgetter(1))
    neighbours1=[]
    for x in range(k):
        neighbours1.append(distances[x][0])
    return neighbours1

def getresponse(neighbours):
    classvotes={}
    for x in range(len(neighbours)):
        response=neighbours[x][-1]
        if response in classvotes:
            classvotes[response]+=1
        else:
            classvotes[response]=1
    sortedvotes=sorted(classvotes.items(),key=operator.itemgetter(1),
                       reverse=True)
   
    return sortedvotes[0][0]
reading()
      


