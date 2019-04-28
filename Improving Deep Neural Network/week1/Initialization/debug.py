import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from init_utils import *


plt.rcParams["figure.figsize"] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_x, train_y, test_x, test_y = load_dataset()


def initialize_parameters_zeros(layers_dims):
    parameters = {}
    L = len(layers_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l - 1]))
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


def initialize_parameters_random(layers_dims):
    parameters = {}
    L = len(layers_dims)
    np.random.seed(3)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 10
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters


def initialize_parameters_he(layers_dims):
    parameters = {}
    L = len(layers_dims)
    np.random.seed(3)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(
            2. / layers_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return parameters

def model(X, Y, learning_rate=0.01, num_iterations=15000, print_cost=True, initialization="he"):
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0], 10, 5, 1]
    parameters = {}

    if initialization == 'zero':
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == 'random':
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)

    for i in range(0, num_iterations):
        a3, cache = forward_propagation(X, parameters)

        cost = compute_loss(a3, Y)

        grads = backward_propagation(X, Y, cache)

        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration{}:{}".format(i, cost))
            costs.append(cost)

    plt.plot(costs)
    plt.xlabel('iterations(per hunderds)')
    plt.ylabel('cost')
    plt.title('Learning rate = ' + str(learning_rate))
    plt.show()

    return parameters

if __name__ == '__main__':

    parameters = model(train_x, train_y, initialization = 'he')
    print("On the train strat")
    prediction_train = predict(train_x, train_y, parameters)
    print("On the test set:")
    prediction_test = predict(test_x, test_y, parameters)
