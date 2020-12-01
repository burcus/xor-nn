import math
import numpy as np

trained = 0
correctPredicted = 0
errorTolerance = 0.7
learnSpeed = 0.6
totalAccuracy = 0
epoch = 1

neuron00weights = [0.01, 0.01] #hidden layer first neuron weights
neuron01weights = [0.02, 0.02] #hidden layer second neuron weights
neuron10weights = [0.01, 0.01] #output layer weights

weights = [neuron00weights, neuron01weights, neuron10weights]

def sigmoid(sum):
  return 1 / (1 + math.exp(-sum))

def sumFunction(inputs, weights):
    neuronSum = 0
    for i in range (0, len(weights), 1):
        neuronSum += inputs[i]*weights[i]
    return neuronSum  

def calculateError(currentResult, correctResult):
    correctResult = float(correctResult)
    error = currentResult*(1-currentResult)*(correctResult-currentResult) #Calculate error     
    return error

def updateWeights(currentResult, error):
    delta = learnSpeed*error*currentResult
    for i in range(0,len(weights),1):
        for j in range(0, len(weights[i]), 1):
            weights[i][j] = weights[i][j] + delta
    
# def updateWeights(currentResult, error):
#     delta = learnSpeed*error*currentResult
#     for i in range(0,len(weights),1):
#         for j in range(0, len(weights[i]), 1):
#             weights[i][j] = weights[i][j] + delta

def updateOutputWeights(currentResult, error):
    delta = learnSpeed*error*currentResult
    for i in range(0,len(weights),1):
        for j in range(0, len(weights[i]), 1):
            weights[i][j] = weights[i][j] + delta

def updateHiddenWeights():


def test(testData):
    for i, neuron in enumerate(hiddenLayer):
        neuronSum = sumFunction(testData, neuron)
        hiddenOuts[i] = sigmoid(neuronSum)
    outputSum = sumFunction(hiddenOuts, neuron10weights)
    output = sigmoid(outputSum)
    error = calculateError(output, correctResult)
    print(f'Tahmin: {output} Hata: {error}')

data = [[1,1], [0,0], [1,0], [0,1]]
results = [0, 0, 1, 1]
hiddenOuts =[[],[]]
hiddenLayer = [weights[0], weights[1]]

for j in range(0, epoch, 1):
    for k, inputs in enumerate(data):
        correctResult = results[k]
        for i, neuron in enumerate(hiddenLayer):
            neuronSum = sumFunction(inputs, neuron)
            hiddenOuts[i] = sigmoid(neuronSum)
            # print(f'Toplam: {neuronSum}')
            # print(f'Out: {hiddenOuts[i]}')
        # print("------------- Toplamlar Hesaplandı -------------")
        # print(f'HIDDEN SUMS: {hiddenOuts}')
        outputSum = sumFunction(hiddenOuts, neuron10weights)
        # print(f'Output Sum: {outputSum}')
        output = sigmoid(outputSum)
        error = calculateError(output, correctResult)
        print(f'Beklenen: {correctResult} Tahmin: {output} Hata: {error}')
        # print(f'Ağırlıklar önce 00: {neuron00weights} 01: {neuron01weights} 10: {neuron10weights}')
        # updateWeights(output, error)
        # print(f'Ağırlıklar sonra 00: {neuron00weights} 01: {neuron01weights} 10: {neuron10weights}')
        # print("-------------  ------------- \n \n \n")

print("Ağ Ağırlıkları Güncellendi")

while True:
    testData = np.array(input("Test Edilecek İkili: ").split(" ")).astype(int)
    print(testData)
    test(testData)

