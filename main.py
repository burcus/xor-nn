import random
import math
import numpy as np
from sklearn.model_selection import KFold
from random import randint


trained = 0
correctPredicted = 0
errorTolerance = 0.7
learnSpeed = 0.6
totalAccuracy = 0

# neuron00weights = np.random.rand(2,1)
# neuron01weights = np.random.rand(2,1)
# neuron10weights = np.random.rand(2,1)
neuron00weights = [1, 3] #hidden layer first neuron weights
neuron01weights = [2, -1] #hidden layer second neuron weights
neuron10weights = [0.3, 0.1] #output layer weights

hiddenLayer = [neuron00weights, neuron01weights]

def sigmoid(sum):
  return 1 / (1 + math.exp(-sum))

def sumFunction(inputs, weights):
    neuronSum = 0
    for i in range (0, len(weights), 1):
        print(inputs[i])
        print(weights[i])
        # neuronSum += inputs[i]*weights[i]
        print(f'Girdi: {inputs[i]}, Çarpılan Ağırlık: {weights[i]}')
    # for i in range(0, len(withoutId),1):
    #     neuronSum = neuronSum + float(withoutId[i])*neuronWeights[i][0]
    return neuronSum  

# def hiddenLayer(dataSetPiece):
#     neuron00 = sumFunction(dataSetPiece, neuron00weights) #Calculate neuron00 Sum
#     derivativeNeuron00 = sigmoid(neuron00) #Get derivative result for neuron00 Sum 
#     neuron01 = sumFunction(dataSetPiece, neuron01weights)
#     derivativeNeuron01 = sigmoid(neuron01)
#     hiddenLayer1(derivativeNeuron00, derivativeNeuron01, dataSetPiece[10]) #Send sums to second hidden layer

def calculateError(currentResult, correctResult):
    correctResult = float(correctResult)
    error = currentResult*(1-currentResult)*(correctResult-currentResult) #Calculate error     
    return error

# def updateWeights(currentResult, error):
#     delta = learnSpeed*error*currentResult
#     for i in range(0,len(weights),1):
#         for weight in weights[i]:
#             weight = weight + delta
         
data = [[1,1,0], [0,0,0], [1,0,1], [0,1,1]]
hiddenSums =[[],[]]

for element in data:
    inputs = [element[0], element[1]]
    correctResult = element[2]
    for i, neuronWeights in enumerate(hiddenLayer):
        hiddenSums[i] = sumFunction(inputs, neuronWeights)
        print(f'Toplam: {hiddenSums[i]}')
    print("------------- Toplamlar Hesaplandı -------------")
    print(f'HIDDEN SUMS: {hiddenSums}')
    outputLayerSum = sumFunction(hiddenSums, neuron10weights)
    print(f'Output Sum: {outputLayerSum}')
    print("-------------  ------------- \n")
    # neuronOutput = sigmoid(hiddenLayerSum) #input layer result
    # error = calculateError(neuronOutput, correctResult)
    # print(f'Beklenen: {correctResult} Tahmin: {neuronOutput} Hata: {error}')
    # print(f'Ağırlıklar önce 00: {neuron00weights} 01: {neuron01weights} 10: {neuron10weights}')
    # updateWeights(neuronOutput, error)
    # print(f'Ağırlıklar sonra 00: {neuron00weights} 01: {neuron01weights} 10: {neuron10weights}')
    # print("\n")


    # for j in range(0, len(inputs), 1):
    #   neuronSum[j] = sumFunction(inputs, weights[j])
