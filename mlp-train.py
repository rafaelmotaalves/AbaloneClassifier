from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

file = open("abalone_formatted.csv")

dataarray = file.readlines()

inputData = []
classData = []

# Formatar valores para entrada do skitlearn
for line in dataarray:
    trimmedLine = "".join(line.split('\n'))
    dataArr=trimmedLine.split(',')
    classData.append(dataArr[8])
    
    a = list(map(float,dataArr[1:8]))

    if(dataArr[0] == 'M'):
        a.append(1)
    elif(dataArr[0] == 'F'):
        a.append(2)
    else:
        a.append(3)
    
    inputData.append(a)

# Configuração do classficador
mlp = MLPClassifier(
    solver='lbfgs',
    learning_rate_init=0.00001
)

mlp.fit(inputData, classData)

for c in mlp.coefs_:
   print(list(c))
