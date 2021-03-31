# from https://gist.github.com/clarkenciel/d9814b298e8f8d134bd4e7d88cda6e48

import numpy as np
from random import choice
import sys


def feed_forward(network, input, activation):
    activations = [input]
    output = input
    
    for layer in network:
        output = activation(np.dot(output, layer))
        activations.append(output)
        
    return output, activations


def calculate_deltas(network, activations, error, derivative):
    """
    :param: network - a neural network
    :param: activations - the activations of the network during the last feed-forward pass
    :param: error - the difference between the target value and the network's final output
    :param: derivative - the derivative of the activation function used in the network's layers
    returns the deltas for each layer of the network. a delta is the amount by which each layer
    missed the value it would have needed to output for the network to match the target value
    """
    deltas = []
    for layer, activation in zip(reversed(network), reversed(activations)):
        delta = error * derivative(activation)
        error = np.dot(delta, layer.T)
        deltas = [delta] + deltas
        
    return deltas


def calculate_updates(activations, deltas):
    """
    :param: activations - the activations of each layer of a neural network during the last feed-forward pass
    :param: deltas - the list of deltas for each layer of the neural network
    
    return updates that should be applied to the weights of the neural network to improve performance.
    
    conceptually this works through the weights of each layer of the network and figures out how much
    and in what direction each weight contributed to the final error.
    """
    return [np.dot(np.transpose(activation), delta)
            for activation, delta in zip(activations, deltas)]        


def apply_updates(network, deltas, learning_rate):
    return [layer + (delta * learning_rate)
            for layer, delta in zip(network, deltas)]        


def train(network, observations, activation, activation_derivative, learning_rate=0.5, limit=100):
    errors = []
    for training_round in range(limit):
        input, target = choice(observations)

        guess, activations = feed_forward(network, input, activation)
        error = target - guess
                
        deltas = calculate_deltas(network, activations, error, activation_derivative)
        updates = calculate_updates(activations, deltas)
        network = apply_updates(network, updates, learning_rate)

        errors.append(error)
        if training_round % (limit * 0.1) == 0:
            error_avg = sum(errors) / len(errors)
            errors = []
            print("Avg error by round {}: {}".format(training_round, error_avg), file=sys.stderr)

    return network


def dtanh(x):
    return 1 - np.power(np.tanh(x), 2)


def test(network, activation, test_set):
    for input, target in test_set:
        guess, _ = feed_forward(network, input, activation)
        print("Guessed {} vs. {} for input {}".format(guess, target, input), file=sys.stderr)
        
if __name__ == "__main__":

    ### learning XOr
    xor_table = [
        (np.array([[0, 1]]), 1),
        (np.array([[1, 0]]), 1),
        (np.array([[1, 1]]), 0),
        (np.array([[0, 0]]), 0),
    ]

    # pretend that we're just observing this phenomenon...
    observations = [choice(xor_table) for _ in range(100)]

    # set up a simple network
    input_size = 2
    hidden_size = 3
    output_size = 1
    network = [
        np.random.random((input_size, hidden_size)),
        np.random.random((hidden_size, output_size))
    ]
    final = train(network, observations, np.tanh, dtanh, limit=100000, learning_rate=0.1)

    test(final, np.tanh, [choice(xor_table) for _ in range(10)])
