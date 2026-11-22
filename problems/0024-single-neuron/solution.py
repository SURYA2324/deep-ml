import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
    probabilities = []
    squared_errors = []
    for features_vector, label in zip(features,labels):
        z = sum(f*w for f,w in zip(features_vector,weights))+bias

        prob = 1/(1+math.exp(-z))
        probabilities.append(round(prob,4))

        squared_errors.append((prob-label)**2)

        mse = round(sum(squared_errors)/len(labels),4)
    return probabilities, mse