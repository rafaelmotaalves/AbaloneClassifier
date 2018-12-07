from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

file = open("abalone_formatted.csv")

dataarray = file.readlines()

inputData = []
classData = []

# Formatar valores para entrada do skitlearn
for line in dataarray:
    trimmedLine = "".join(line.split('\n'))
    dataArr=trimmedLine.split(',')
    classData.append(dataArr[8])
    
    a = list(map(float,dataArr[1:7]))

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


# Não foi possível otimizar a accuracy regulando a quantidade de neurons
# O melhor algoritmo de ativação foi relu
scores = cross_val_score(mlp, inputData, classData, cv=10)
print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))