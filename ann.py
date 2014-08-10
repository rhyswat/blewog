#!/usr/bin/env python3

import collections
import itertools
import math
import pprint
import random

class ANN:
    '''
    Artifiial Neural Network.

    This implementation is based on https://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf
    The data structures don't use Nodes directly. Instead each node is represented by
    an integer nymber; each layer is a list of the numbers of the nodes in that layer.
    
    This implementation uses the sigmoid transfer function t(z) = 1/(1+exp(-z))
    '''
    def __init__(self, *sizes, seed=None):
        '''
        Pass in a list of integers:
          number of nodes in input layer, num in next layer, ..., num in output layer
        '''
        self.learning_rate = 0.95 # 0 < l
        self.rand = random.Random(seed)

        self.outputs = dict()  # neuron number => activation
        self.weights = dict()  # digraph of index(number-from, number-to) => weight
        self.deltas = dict()   # neuron number => its error
        self.targets = dict()  # output neuron number => what its activation should be
        c = itertools.count(1) # node number generator
        self.layers = []      # list of neuron numbers in each layer

        for layer_size in sizes:
            this_layer = []
            for n in range(layer_size):
                num = next(c)
                this_layer.append(num)
                self.outputs[num] = 0
                self.deltas[num] = 0
                self.targets[num] = 0
                if len(self.layers) > 0:
                    # not the input layer -> create randomised initial weights
                    for _from in self.layers[-1]:
                        self.set_weight(_from, num, self.rand.uniform(-1, 1))
            self.layers.append(this_layer)

    # --------------------------------------------------------------------------
    # Pairing functions for weights
    # There are lots of strategies:
    # - Cantor pairing
    # - ....
    # - bit packing, e.g. 2 4-byte ints into a 8-byte long
    # See http://stackoverflow.com/questions/919612/mapping-two-integers-to-one-in-a-unique-and-deterministic-way
    # and http://en.wikipedia.org/wiki/Pairing_function
    
    # pack two non-negative (but signed) ints into a long
    def index(self, a, b):
        return a << 16 | b

    # get the weight value for the link i -> j
    def get_weight(self, i, j):
        return self.weights[self.index(i,j)]
    
    # set the weight value for the link i -> j to w
    def set_weight(self, i, j, w):
        self.weights[self.index(i,j)] = w

    # increment the weight value for the link i -> j by dw
    def add_weight(self, i, j, dw):
        self.weights[self.index(i,j)] += dw

    # get a list of node numbers where weight(*, num) exists
    def links_to(self, num):
        return [x >> 16 for x in filter(lambda x : x & 0xFFFF == num, self.weights.keys())]
 
    # get a list of node numbers where weight(num, *) exists
    def links_from(self, num):
        return [x & 0xFFFF for x in filter(lambda x : x >> 16 == num, self.weights.keys())]

    # --------------------------------------------------------------------------
    # Sigmoid transfer function t(z), dt/dz = t(z)(1-t(z))
    def transfer(self, z):
        return 1/(1 + math.exp(-z))

    # dt/dz = t(z)(1-t(z)
    def transfer_prime(self, t):
        return t*(1-t)

    # feed-forward from input layer to output layer
    def feed_forward(self, inputs):
        # set the inputs
        for i in range(len(self.layers[0])):
            self.outputs[self.layers[0][i]] = inputs[i]
             
        # feed forward
        for _layer in range(1, len(self.layers)):
            for _to in self.layers[_layer]:
                s = sum(self.outputs[_from] * self.get_weight(_from, _to) for _from in self.layers[_layer-1])
                self.outputs[_to] = self.transfer(s)

    # Back-propagation from the output to the input
    def back_propagate(self, expected_output):
        # calculate the output-layer deltas
        for i in range(len(expected_output)):
            num = self.layers[-1][i]
            self.targets[num] = expected_output[i]
            a = self.outputs[num]
            self.deltas[num] = self.transfer_prime(a)*(self.targets[num]-a)

        # update weights into the output layer
        for j in self.layers[-1]:
            for i in self.links_to(j):
                self.add_weight(i, j, self.learning_rate*self.deltas[j]*self.outputs[i])
            
        # now back-propagate from layer last-but-one
        for _layer_index in range(len(self.layers)-2, -1, -1):
            this_layer = self.layers[_layer_index]
            for num in this_layer:
                a = self.outputs[num]
                self.deltas[num] = a*(1-a)*sum(self.deltas[j]*self.get_weight(num, j) for j in self.links_from(num))
            
            for num in this_layer:
                for _into_num in self.links_to(num):
                    self.add_weight(_into_num, num, self.learning_rate*self.deltas[num]*self.outputs[_into_num])

    # feed-forward and back-propagate the data once only
    def train(self, inputs, expected_outputs):
        self.feed_forward(inputs)
        self.back_propagate(expected_outputs)

    # feed the input through the network and return the activations of the output layer
    def evaluate(self, inputs):
        self.feed_forward(inputs)
        return list(self.outputs[x] for x in self.layers[-1])

    # This is the sum-squared of all errors in the network
    def total_error(self):
        return sum(x**2 for x in self.deltas.values())
        
    # Some debug
    def show(self):
        print('Layer indices')
        pprint.pprint(self.layers)
        print('Activations')
        pprint.pprint(self.outputs)
        print('Errors')
        pprint.pprint(self.deltas)
        print('Weights')
        unpair = lambda x : (x >> 16, x & 0xFFFF)
        w = [(unpair(kv[0]), kv[0], kv[1]) for kv in self.weights.items()]
        w.sort(key=lambda x :x[0])
        pprint.pprint(w)
# ------------------------------------------------------------------------------

def learn_xor():
    # try learning the XOR function f(x, y) = x XOR y
    # input layer = 2 elements [x, y] which will be 0 or 1
    # output layer = 1 element which should be x XOR y
    # arbitraray choice of hidden layers
    ann = ANN(2, 4, 1)
    ann.show()
    for i in range(1000):
        ann.train([0, 0], [0])
        ann.train([0, 1], [1])
        ann.train([1, 0], [1])
        ann.train([1, 1], [0])
        if i > 100 and ann.total_error() < 1e-4: # occasionally fails whatever this is set to
            print('Breaking early at ' + str(i+1))
            break
    print('Total error = {}'.format(ann.total_error()))
    for y in [0, 1]:
        for x in [0, 1]:
            xy = (x, y)
            # 0<=evaluation<=0, so threshold at 0.5
            e = ann.evaluate(xy)
            if e[0] > 0.5:
                result = 1
            else:
                result = 0
            print('Testing {}: expected={}, actual={}'.format(xy, x ^ y, result))
    
if __name__ == '__main__':
    learn_xor()
