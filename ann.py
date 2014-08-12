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
    
    This implementation uses the sigmoid transfer function t(z) = 1/(1+exp(-z)).

    Updates should be applied sequentially, so for example one epoch of training
    looks like this:
    
    >>> ann = ANN(2, 3, 4) # for example
    >>> for input, output in training_set:
    ...   ann.train(input, output)

    Multiple epochs should shuffle the training data between presentations to combat
    over-fitting, thus:

    >>> import random
    >>> ann = ANN(2, 3, 4) # for example
    >>> for n in range(num_epochs) # several hundred is a good start
    ...    for input, output in training_set:
    ...       ann.train(input, output)
    ...    random.shuffle(training_set)

    Shuffling seems to lead to faster convergence as well.

    The error in the network is implemented as the sum of the squares of the most
    recently trained error at each node in the network. The user may choose to stop
    the epoch loop when (a) the error gets below some threshold, or (b) when the error
    starts to *increase* compared with the previous epoch. The latter case may be a
    symptom of climbing out of a local minimum in weight space.

    Inputs should be normalised to so that their mean over the training set is close to zero.
    (Haykyn, 1999, pp. 181-182).
    
    The output should be limited to being within the range of the activation function,
    which in this case is (-1, +1) and ideally in (-1 + epsilon, 1 - epsilon) for some
    small epsilon > 0 to avoid saturation (Haykyn, 1999, p. 181).

    These last two points are actually a problem for the _user_ of this class,
    not the ANN itself.
    '''
    def __init__(self, *sizes, seed=None, learning_rate=0.75, momentum=0.1):
        '''
        Pass in a list of integers:
          number of nodes in input layer, num in next layer, ..., num in output layer

        The learning rate should satisfy 0 < rate, and in practice 0.05 <= rate < 0.8
        The momentum should satisfy 0 <= mom < 1. If >= 1 then the momentum causes the
        weight sum to oscillate (Haykyn, 1999, pp. 169-171)
        '''
        self.learning_rate = learning_rate # 0 < rate <= 1
        self.momentum = momentum # 0 < mom < 1
        self.rand = random.Random(seed)

        self.outputs = dict()  # neuron number => activation
        self.weights = dict()  # digraph of index(number-from, number-to) => weight
        self.deltas = dict()   # neuron number => its error
        self.old_deltas = collections.defaultdict(float)
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
                        self.set_weight(_from, num, self.rand.uniform(-0.5, 0.5))
            self.layers.append(this_layer)

    # --------------------------------------------------------------------------
    # Pairing functions for weights
    # There are lots of strategies:
    # - Cantor pairing
    # - Szudzik's function
    # - bit packing two N-byte values into one 2N-byte value (which this class does)
    # See http://stackoverflow.com/questions/919612/mapping-two-integers-to-one-in-a-unique-and-deterministic-way
    # and http://en.wikipedia.org/wiki/Pairing_function
    
    # pack two non-negative integers into one
    # this limits the total number of nodes to 2^16-1 = 65535
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
        self.weights[self.index(i,j)] += (dw + self.momentum*self.old_deltas[self.index(i,j)])

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
        self.old_deltas = collections.defaultdict(float, self.deltas)

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
    rnd = random.Random(76545)
    training = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]
    ann = ANN(2, 3, 1, momentum=0.1)
    for i in range(1000):
        for _in, _out in training:
            ann.train(_in, _out)
        rnd.shuffle(training) # Big help!
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

# the infamous iris data set
# https://archive.ics.uci.edu/ml/datasets/Iris
# corrections applied to 35th and 38th samples
def learn_irises():
    import os.path
    # One row of data in the iris data set
    class Datum(list):

        def __init__(self, f1, f2, f3, f4, label):
            super(list, self).__init__()
            self.extend([f1, f2, f3, f4, label])

        def label(self):
            return self[-1]

        def as_input(self):
            return self[:4]
        
        def as_output(self, classes):
            out = [0]*len(classes)
            i = classes.index(self.label())
            out[i] = 1.0
            return out            

        def normalise(self, feature_index, _min, _max):
            self[feature_index] = (self[feature_index] - _min) / (_max - _min)

        def __hash__(self):
            return hash(tuple(self))

    # read it all into a flat list
    all_data = []
    with open(os.path.join(os.path.dirname(__file__), 'iris.data'), 'rt') as f:
        for x in f:
            items = x.strip().split(',')
            if len(items) == 5:
                ff = [float(y) for y in items[:-1]] + [items[-1].strip()]
                all_data.append(Datum(*ff))
    print('I have loaded', len(all_data), 'items')

    # normalisation: feature_n now lies between 0 and 1 inclusive
    minima = [min(all_data, key=lambda x:x[i]) for i in range(4)]
    minima = [minima[i][i] for i in range(4)]
    maxima = [max(all_data, key=lambda x:x[i]) for i in range(4)]
    maxima = [maxima[i][i] for i in range(4)]
    for d in all_data:
        for i in range(4):
            d.normalise(i, minima[i], maxima[i])

    # partition into classes
    by_class = collections.defaultdict(list)
    for d in all_data:
        by_class[d.label()].append(d)
    classes = list(by_class.keys())

    # partition into a training and test set 'at random'
    rnd = random.Random(6543)
    train_pct = 20 # percent of data that go into the training set
    training = []
    test = []
    for label, datum_list in by_class.items():
        n = (len(datum_list) * train_pct) // 100
        s = rnd.sample(range(len(datum_list)), n)
        for i in range(len(datum_list)):
            if i in s:
                training.append(datum_list[i])
            else:
                test.append(datum_list[i])
    print('Partitioned into {} training items and {} test items'.format(len(training), len(test)))
 
    # make and train the ann
    ann = ANN(4, 6, 3, len(classes), seed=9090, momentum=0.05, learning_rate=0.5)
    e1 = 0
    e2 = float('Infinity')
    for epoch in range(1000):
        if (epoch + 1) % 50 == 0:
            print('Epoch {}...'.format(epoch+1))
        for t in training:
            ann.train(t.as_input(), t.as_output(classes))
        e1 = ann.total_error()
        if e2 < 1e-7 and e1 > e2: # last bit tries to avoid overfitting
            print('Breaking at epoch {} with error {}'.format(epoch+1, e2))
            break
        e2 = e1
        rnd.shuffle(training) # present in a different order @ next epoch
    print('Trained the thing, error is {}'.format(ann.total_error()))

    # helper that takes the output [x, y, z] and returns the class
    # corresponding to the max of these.
    def to_class(x):
        z = max(x)
        i = x.index(z)
        return classes[i]

    # test it
    correct = collections.defaultdict(int)
    for c in classes:
        correct[c] = 0
    test_classes = collections.defaultdict(list)
    for t in test:
        test_classes[t.label()].append(t)
    for t in test:
        o = ann.evaluate(t.as_input())
        a = to_class(o)
        if t.label() == a:
            correct[t.label()] += 1
    print('Results:')
    for clz, count in correct.items():
        print('Class {}, {:.2f}% correct out of {}'.format(clz, 100.0*count / len(test_classes[clz]), len(test_classes[clz])))

if __name__ == '__main__':
    print()
    print(' -- XOR --')
    learn_xor()

    print()
    print('-- IRISES --')
    learn_irises()
