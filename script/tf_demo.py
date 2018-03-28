# Trying to define the simplest possible neural net where the output layer of the neural net is a single
# neuron with a "continuous" (a.k.a floating point) output.  I want the neural net to output a continuous
# value based off one or more continuous inputs.  My real problem is more complex, but this is the simplest
# representation of it for explaining my issue.  Even though I've oversimplified this to look like a simple
# linear regression problem (y=m*x), I want to apply this to more complex neural nets.  But if I can't get
# it working with this simple problem, then I won't get it working for anything more complex.

import tensorflow as tf
import random
import numpy as np
import networkx

INPUT_DIMENSION = 1
OUTPUT_DIMENSION = 1
TRAINING_RUNS = 100
BATCH_SIZE = 10000
VERF_SIZE = 1

# load training data
edges = pd.read_table("~/Desktop/Project/data/smallGraph.txt",
                    sep = " ",
                    header = None, 
                    names = ['vx', 'vy', 'weight'])

graph = nx.from_pandas_edgelist(edges, 'vx', 'vy', 'weight')
# graph_nodes = graph.nodes()
graph_dict = nx.to_dict_of_dicts(graph)
G = nx.Graph(graph_dict)
distanceMatrix = np.load("smallGraph_distanceMatrix.dat")

# Generate two arrays, the first array being the inputs that need trained on, and the second array containing outputs.
def generate_test_point():
    x = random.uniform(-8, 8)

    # To keep it simple, output is just -x.
    out = -x

    return (np.array([x]), np.array([out]))


# Generate a bunch of data points and then package them up in the array format needed by
# tensorflow
def generate_batch_data(num):
    xs = []
    ys = []

    for i in range(num):
        x, y = generate_test_point()

        xs.append(x)
        ys.append(y)

    return (np.array(xs), np.array(ys))


# Define a single-layer neural net.  Originally based off the tensorflow mnist for beginners tutorial

# Create a placeholder for our input variable
x = tf.placeholder(tf.float32, [None, INPUT_DIMENSION])

# Create variables for our neural net weights and bias
W = tf.Variable(tf.zeros([INPUT_DIMENSION, OUTPUT_DIMENSION]))
b = tf.Variable(tf.zeros([OUTPUT_DIMENSION]))

# Define the neural net.  Note that since I'm not trying to classify digits as in the tensorflow mnist
# tutorial, I have removed the softmax op.  My expectation is that 'net' will return a floating point
# value.
net = tf.matmul(x, W) + b

# Create a placeholder for the expected result during training
expected = tf.placeholder(tf.float32, [None, OUTPUT_DIMENSION])

# Same training as used in mnist example
loss = tf.reduce_mean(tf.square(expected - net))
# cross_entropy = -tf.reduce_sum(expected*tf.log(tf.clip_by_value(net,1e-10,1.0)))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)