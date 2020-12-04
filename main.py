import math
import numpy as np

# trained = 0
# correctPredicted = 0
# errorTolerance = 0.7
learnSpeed = 0.6
totalAccuracy = 0
epoch = 1

# neuron00weights = [0.0, 0.1] #hidden layer first neuron weights
# neuron01weights = [1.0, 1.1] #hidden layer second neuron weights
# neuron10weights = [2.0, 2.1] #output layer weights

neuron00weights = [1, 1] #hidden layer first neuron weights
neuron01weights = [-1, -1] #hidden layer second neuron weights
neuron10weights = [1, 1] #output layer weights

weights = [neuron00weights, neuron01weights, neuron10weights]
outputLayerWeights = neuron10weights
hiddenLayerWeights =[neuron00weights, neuron01weights]

def sigmoid(sum):
  return 1 / (1 + math.exp(-sum))

def sumFunction(inputs, weights):
    neuronSum = 0
    for i in range (0, len(weights), 1):
        neuronSum += inputs[i]*weights[i]
    return neuronSum  

def outputError(currentResult, correctResult):
    correctResult = float(correctResult)
    # error = 1/2*math.pow((correctResult - currentResult), 2)
    error = currentResult*(1-currentResult)*(correctResult-currentResult) #Calculate error     
    return error

def hiddenError(currentResult, correctResult, error, Wj5):
    error = currentResult*(1-currentResult)*error*Wj5
    return error

def updateOutputWeights(error):
    for i in range(0, len(hiddenOuts), 1):
        delta = learnSpeed*error*hiddenOuts[i]
        outputLayerWeights[i] = outputLayerWeights[i] + delta

def updateHiddenWeights(hiddenOuts, hiddenErrors, inputs):
    for i in range(0, len(hiddenErrors), 1):
        for j in range(0, len(hiddenLayerWeights), 1):
            delta = hiddenErrors[i]*learnSpeed*inputs[j]
            hiddenLayerWeights[i][j] = hiddenLayerWeights[i][j] + delta

def test(testData):
    for i, neuron in enumerate(hiddenLayerWeights):
        neuronSum = sumFunction(testData, neuron)
        hiddenOuts[i] = sigmoid(neuronSum)
    outputSum = sumFunction(hiddenOuts, neuron10weights)
    output = sigmoid(outputSum)
    error = outputError(output, correctResult)
    print(f'Prediction: {output} Error: {error}')

data = [[1,1], [0,0], [1,0], [0,1]]
results = [0, 0, 1, 1]

# data = [[1,1], [0,0], [1,0], [0,1], [1,1], [0,0], [1,1], [0,0], [1,0], [0,1], [1,1], [0,0], [1,1], [0,0], [1,0], [0,1], [1,1], [0,0], [1,1], [0,0], [1,0], [0,1], [1,1], [0,0], [1, 0], [1, 1], [1, 1], [0, 0], [1, 0], [0, 1]]
# results = [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1,0,0,0,1,1]

hiddenOuts =[[],[]]
hiddenSums = [[],[]]
hiddenErrors = [[],[]]

# print(weights)

for j in range(0, epoch, 1):
    for k, inputs in enumerate(data):
        correctResult = results[k]
        for i, neuron in enumerate(hiddenLayerWeights):
            neuronSum = sumFunction(inputs, neuron)
            hiddenSums[i] = neuronSum
            hiddenOuts[i] = sigmoid(neuronSum)
        outputSum = sumFunction(hiddenOuts, outputLayerWeights)
        output = sigmoid(outputSum)
        error = outputError(output, correctResult)
        for i, neuron in enumerate(hiddenLayerWeights):
            hiddenErrors[i] = hiddenError(hiddenOuts[i], correctResult, error, outputLayerWeights[i])
        print(f'Result: {correctResult} Prediction: {output} Error: {error}')
        updateHiddenWeights(hiddenOuts, hiddenErrors, inputs)
        updateOutputWeights(error)

print("Weights updated")
# print(weights)

while True:
    testData = np.array(input("Test Values: ").split(" ")).astype(int)
    test(testData)

