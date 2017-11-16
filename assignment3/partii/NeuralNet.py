#########################################################
######                 NeuralNet                    #####
#########################################################
# python NeuralNet.py -i data/iris_processed.csv -p 0.8 -m 200 -d 4 -n 4


import argparse
import pandas as pd
from random import random
from math import exp
import numpy as np

parser = argparse.ArgumentParser(description='Neural Net.')
parser.add_argument('-i', '--input', help='Path of input dataset')
parser.add_argument('-p', '--train_percent', help='percentage of the dataset to be used for training')
parser.add_argument('-m', '--max_iter', help='Maximum number of iterations that your algorithm will run')
parser.add_argument('-d', '--hidden', help='number of hidden layers')
parser.add_argument('-n', '--neuron', help='number of neurons in each hidden layer')

args = parser.parse_args()
data_path = args.input
train_percent = float(args.train_percent)
max_iter = int(args.max_iter)
n_hidden = int(args.hidden)
n_neuron = int(args.neuron)

# print(data_path, train_percent, max_iter, n_hidden, n_neuron)

def multi_layer_network(n_inputs, n_hidden, n_neuron_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_neuron_hidden)]
    network.append(hidden_layer)
    if n_hidden > 1:
        for i in range(n_hidden-1):
            hidden_layer = [{'weights': [random() for i in range(len(hidden_layer) + 1)]} for i in range(n_neuron_hidden)]
            network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_neuron_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network

def activation_func(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation

def sigmoid(activation):
    return 1.0 / (1.0 + exp(-activation))

def forward_propagate(network, input_layer):
    inputs = input_layer
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activation_func(neuron['weights'], inputs)
            neuron['output'] = sigmoid(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

def sigmoid_prime(output):
    return output * (1.0 - output)

def backward_propagate(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * sigmoid_prime(neuron['output'])

def update_weights(network, input, eta):
    for i in range(len(network)):
        inputs = input[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += eta * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += eta * neuron['delta']

def train_neuralNet(network, data, eta, n_iter, n_outputs):
    for iter in range(n_iter):
        sum_error = 0
        for instance in data:
            outputs = forward_propagate(network, instance)
            expected = [0 for i in range(n_outputs)]
            expected[int(instance[-1])] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate(network, expected)
            update_weights(network, instance, eta)
        # print('# iter=%d, eta=%.3f, error=%.3f' % (iter, eta, sum_error))

def predict(network, instance):
    outputs = forward_propagate(network, instance)
    return outputs.index(max(outputs))

def train_predict_neuralNet(train_data, test_data, eta, n_iter, n_hidden, n_neuron_hidden):
    n_inputs = len(train_data[0]) - 1
    n_outputs = len(set([instance[-1] for instance in train_data]))
    network = multi_layer_network(n_inputs=n_inputs, n_hidden=n_hidden, n_neuron_hidden=n_neuron_hidden, n_outputs=n_outputs)
    train_neuralNet(network, data=train_data, eta=eta, n_iter=n_iter, n_outputs=n_outputs)
    predictions_train = list()
    for instance in train_data:
        prediction = predict(network, instance)
        predictions_train.append(prediction)
    predictions_test = list()
    for instance in test_data:
        prediction = predict(network, instance)
        predictions_test.append(prediction)
    return network, predictions_train, predictions_test

def evaluate_nerualNet(data_path, train_percent, eta, n_iter, n_hidden, n_neuron_hidden, print_weight=False):
    data = pd.read_csv(data_path)
    data[data.columns[-1]] = data[data.columns[-1]].astype(int)
    data = data.values.tolist()
    n_train = int(len(data) * train_percent)
    idx_train = list(np.random.choice(len(data), n_train, replace=False))
    idx_test = [i for i in range(len(data)) if i not in idx_train]
    train_data = [data[i] for i in idx_train]
    test_data = [data[i] for i in idx_test]
    ### neural network
    network, pred_train, pred_test = train_predict_neuralNet(train_data=train_data, test_data=test_data, eta=eta, n_iter=n_iter, n_hidden=n_hidden, n_neuron_hidden=n_neuron_hidden)
    accuracy_train = cal_accuracy(predicted=pred_train, expected=[instance[-1] for instance in train_data])
    accuracy_test = cal_accuracy(predicted=pred_test, expected=[instance[-1] for instance in test_data])
    if print_weight is True:
        for i, layer in enumerate(network):
            print('\nLayer {}:'.format(i))
            for j, neuron in enumerate(layer):
                print('\tNeural {} weight: '.format(j), [round(weight, 3) for weight in neuron['weights']])
    print('\nTotal training error = {}%'.format(accuracy_train))
    print('Total test error = {}%'.format(accuracy_test))

def cal_accuracy(expected, predicted):
    correct = 0
    for i in range(len(expected)):
        if expected[i] == predicted[i]:
            correct += 1
    return correct / float(len(expected)) * 100.0

evaluate_nerualNet(data_path=data_path, train_percent=train_percent, eta=0.5, n_iter=max_iter, n_hidden=n_hidden, n_neuron_hidden=n_neuron, print_weight=True)