# Test environment for optimizing the fitness function in COTONN
# and testing some tensorboard functionality

# imports
import tensorflow as tf 
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
x = tf.placeholder(tf.float32, [None, dataSet.getXDim()], "X-Data")
y = tf.placeholder(tf.float32, [None, dataSet.getYDim()], "Y-Data")

with tf.name_scope("Hidden1"):
    hidden1 = tf.layers.dense(inputs=x, units=6, activation=tf.sigmoid, name="Dense1")
    hidden1 = tf.layers.dropout(inputs=hidden1, rate=0.05)

with tf.name_scope("Hidden2"):
    hidden2 = tf.layers.dense(inputs=hidden1, units=6, activation=tf.sigmoid, name="Dense2")
    hidden2 = tf.layers.dropout(inputs=hidden2, rate=0.05)
    
with tf.name_scope("Hidden3"):
    hidden3 = tf.layers.dense(inputs=hidden2, units=6, activation=tf.sigmoid, name="Dense3")
    hidden3 = tf.layers.dropout(inputs=hidden3, rate=0.05)

with tf.name_scope("Estimation"):
    estimation = tf.layers.dense(inputs=hidden3, units=dataSet.getYDim(), activation=tf.sigmoid, name="Estimation")

# setup loss function
with tf.name_scope("Loss"):
    #loss = tf.losses.softmax_cross_entropy(y, estimation)
    loss = tf.losses.log_loss(y, estimation)
    tf.summary.scalar("loss", loss)

# setup training function
with tf.name_scope("Train"):
    train = tf.train.AdamOptimizer(0.005).minimize(loss)

# setup fitness function
with tf.name_scope("Fitness"):
    eta = [-0.5, 0.5]
    lower_bound = tf.add(y, eta[0])
    upper_bound = tf.add(y, eta[1])

    fit = tf.logical_and(tf.greater_equal(estimation, lower_bound), tf.less(estimation, upper_bound))
    non_zero = tf.to_float(tf.count_nonzero(fit))
    amount = tf.to_float(tf.size(fit))
    fitness = non_zero/amount
    tf.summary.scalar("fitness", fitness)
    
# tensorboard
writer = tf.summary.FileWriter("test", session.graph)
merged = tf.summary.merge_all()
    
# initialize all global variables
session.run(tf.global_variables_initializer())

# train
epoch, batch_index, batch_size, display_step, i, l, f = 0, 0, 250, 1000, 0, 0.0, 0.0
size = dataSet.getSize()

while epoch <= 25000:
    batch = dataSet.getBatch(batch_size, batch_index)
    session.run(train, {x: batch[0], y: batch[1]})
    
    if(i % display_step == 0 and i != 0):
        l = session.run(loss, {x: batch[0], y: batch[1]})
        f = session.run(fitness, {x: dataSet.x, y: dataSet.y})
        summary = session.run(merged, {x: batch[0], y: batch[1]})
        print("i = " + str(i) + "\tepoch = " + str(epoch) + "\tloss = " + str(float("{0:.3f}".format(l))) + " fitness = " + str(float("{0:.3f}".format(f))))
        writer.add_summary(summary, i)
        
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
