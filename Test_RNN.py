# Test environment for optimizing the fitness function in COTONN
# and testing some tensorboard functionality

# imports
import tensorflow as tf 
from tensorflow.contrib import rnn
from DataSet import DataSet
from Importer import Importer
from StaticController import StaticController

print("COTONN test environment\n")

# initialize session 
tf.reset_default_graph() # prevents graph overloading from previous session
session = tf.Session()

# importer for file io
importer = Importer()

# read controller
controller = StaticController()
controller = importer.readStaticController("controllers/dcdc/controller")

# initialize and format data set
dataSet = DataSet()
dataSet.readSetFromController(controller)
dataSet.formatToBinary()


# setup graph
x = tf.placeholder(tf.float32, [None, dataSet.getXDim()])
y = tf.placeholder(tf.float32, [None, dataSet.getYDim()])

num_input = dataSet.getXDim() # MNIST data input (img shape: 28*28)
timesteps = 1 # timesteps
num_hidden = 128 # hidden layer num of features
num_classes = dataSet.getYDim() # MNIST total classes (0-9 digits)

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

lstm_cell = rnn.BasicRNNCell(num_hidden, activation=tf.tanh, reuse=None)
outputs, states = rnn.static_rnn(lstm_cell, [x], dtype=tf.float32)

logits = tf.matmul(outputs[-1], weights['out']) + biases['out']
prediction = tf.nn.softmax(logits)

# setup loss function
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
loss = tf.losses.log_loss(y, prediction)
# setup training function
train = tf.train.AdamOptimizer(0.001).minimize(loss)

# Evaluate model (with test logits, for dropout to be disabled)
#correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


eta = [-0.5, 0.5]
lower_bound = tf.add(y, eta[0])
upper_bound = tf.add(y, eta[1])

fit = tf.logical_and(tf.greater_equal(prediction, lower_bound), tf.less(prediction, upper_bound))
non_zero = tf.to_float(tf.count_nonzero(fit))
amount = tf.to_float(tf.size(fit))
fitness = non_zero/amount
    
# initialize all global variables
session.run(tf.global_variables_initializer())

# train
epoch, batch_index, batch_size, display_step, i, l, f = 0, 0, 500, 2000, 0, 0.0, 0.0
size = dataSet.getSize()

while epoch <= 25000:
    batch = dataSet.getBatch(batch_size, batch_index)
    # batch_x = batch[0]
    # batch[0] = batch_x.reshape((batch_size, num_input))
    session.run(train, {x: batch[0], y: batch[1]})
    
    if(i % display_step == 0 and i != 0):
        l = session.run(loss, {x: batch[0], y: batch[1]})
        f = session.run(fitness, {x: batch[0], y: batch[1]})
        print("i = " + str(i) + "\tepoch = " + str(epoch) + "\tloss = " + str(float("{0:.3f}".format(l))) + " fitness = " + str(float("{0:.3f}".format(f))))
        
    if(f >= 1.0):
        print("Fitness reached 100%")
        break
    
    batch_index += batch_size
    if(batch_index >= size):
        batch_index = batch_index % size
        epoch += 1
    
    i += 1

#validate
batch = dataSet.getBatch(1,1)
#print(batch[1])
#print(session.run(estimation, {x: batch[0], y: batch[1]}))

# crude format of the network    
with tf.variable_scope("Dense1", reuse=True):
    print("W1:")
    print(session.run(tf.get_variable("kernel")))
    print("b1:")
    print(session.run(tf.get_variable("bias")))
        
with tf.variable_scope("Dense2", reuse=True):
    print("W2:")
    print(session.run(tf.get_variable("kernel")))
    print("b2:")
    print(session.run(tf.get_variable("bias")))
    
with tf.variable_scope("Dense3", reuse=True):
    print("W3:")
    print(session.run(tf.get_variable("kernel")))
    print("b3:")
    print(session.run(tf.get_variable("bias")))

session.close()
