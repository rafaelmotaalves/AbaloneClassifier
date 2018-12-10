from sklearn.mixture import GaussianMixture
import numpy as np
import sys

file = open("../abalone_formatted.csv")
dataarray = file.readlines()

inputData = []
classData = []

# Formatar valores para entrada do skitlearn
for line in dataarray:
    trimmedLine = "".join(line.split('\n'))
    dataArr=trimmedLine.split(',')
    
    a = list(map(float,dataArr[1:8]))

    if(dataArr[0] == 'M'):
        a.append(1)
    elif(dataArr[0] == 'F'):
        a.append(2)
    else:
        a.append(3)
    
    if(dataArr[8] == 'Young'):
        classData.append(0)
    if(dataArr[8] == 'Adult'):
        classData.append(1)
    if(dataArr[8] == 'Old'):
        classData.append(2)

    inputData.append(a)

line = sys.argv[1]
trimmedLine = "".join(line.split('\n'))
dataArr=trimmedLine.split(',')
    
a = list(map(float,dataArr[1:8]))

if(dataArr[0] == 'M'):
    a.append(1)
elif(dataArr[0] == 'F'):
   a.append(2)
else:
   a.append(3)

p = np.array(a)
gmm = GaussianMixture(n_components=2, covariance_type="full", max_iter=500).fit(inputData)
cluster = gmm.predict(p.reshape(1, -1))

print(cluster[0])