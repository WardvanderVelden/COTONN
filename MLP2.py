import tensorflow as tf
import NeuralNetworkManager
import random

# The MLP class as presented in this file is a bare bone object oriented approach to creating an MLP
# for the purpose of capturing the behaviour of a static controller (table). 
class MLP:   
        def __init__(self):
            # General parameters
            self.learning_rate = 0.1
            self.num_steps = 50
            self.batch_size = 256
            self.display_step = 200
            
            # Network parameters
            self.layers = []
            self.num_layers = 0
            
            # Training and activation parameters
            self.activation_type = None
            self.loss_function = None
            self.optimizer_type = None
            
            # Neural network type specific variables (MLP)
            self.s = None # state tensor
            self.u = None # input tensor
            
            self.weights = []
            self.biases = []
            
            self.network = None # Will contain the actual neural network model
            
        # Setters
        def setLearningRate(self, value): self.learning_rate = value
        def setNumSteps(self, value): self.num_steps = value
        def setBatchSize(self, value): self.batch_size = value
        def setDisplayStep(self, value): self.display_step = value
        def setActivationType(self, value): self.activation_type = value
        def setLossFunction(self, value): self.loss_function = value
        def setOptimizer(self, value): self.optimizer_type = value
      
        def setNeurons(self, layers):
            self.num_layers = len(layers)
            self.layers = layers
      
        # Getters
        def getLearningRate(self): return self.learning_rate
        def getNumSteps(self): return self.num_steps
        def getBatchSize(self): return self.batch_size
        def getDisplayStep(self): return self.display_step
        def getNumLayers(self): return self.num_layers
        def getLayers(self): return self.layers
        def getActivationType(self): return self.activation_type 
        def getOptimizer(self): return self.optimizer_type
        
        # Initialize network function which intializes an initial network with random weights and biases
        def initializeNetwork(self):
            # Initialize tensors
            self.x = tf.Variable("float", [None, self.layers[0]])
            self.y = tf.Variable("float", [None, self.layers[-1]])
            
            self.weights = [None]*(self.num_layers - 1)
            self.biases = [None]*(self.num_layers - 1)
            
            for i in range(self.num_layers - 1):
                self.weights[i] = tf.Variable(tf.random_normal([self.layers[i], self.layers[i+1]]),tf.float32)
                self.biases[i]= tf.Variable(tf.random_normal([self.layers[i+1]]),tf.float32)
                
            # Input layer
            tf_layers = [None]*(self.num_layers - 1)
            
            print(self.weights[1])
            print(self.biases[1])
            
            tf_layers[0] = tf.add(tf.matmul(self.x, self.weights[0]), self.biases[0]) # input layer
            for i in range(1, self.num_layers - 1):
                tf_layers[i] = tf.add(tf.matmul(tf_layers[i-1], self.weights[i]), self.biases[i])
            self.network = tf_layers[-1]
            return self.network
        
        # TensorFlow MLP test with an extremely simple network with only 1 layer in between to get my head around tf
        def tensorFlowTest(self, nnm):
            # instantiate an interactive tensorflow session 
            sess = tf.InteractiveSession()
            
            # placeholder tensors into which the batches will be loaded
            self.s = tf.placeholder(tf.float32, shape=[None, self.layers[0]])
            self.u = tf.placeholder(tf.float32, shape=[None, self.layers[-1]]) 
            
            # tensors which contain the weights and biases (weight matrices and bias vectors)
            self.weights = [None]*2
            self.biases = [None]*2
            
            self.weights[0] = tf.Variable(tf.random_normal([self.layers[0], self.layers[1]]))
            self.biases[0] = tf.Variable(tf.random_normal([self.layers[1]]))
            
            self.weights[1]= tf.Variable(tf.random_normal([self.layers[1], self.layers[-1]]))
            self.biases[1] = tf.Variable(tf.random_normal([self.layers[-1]]))
            
            # initialize variables    
            sess.run(tf.global_variables_initializer())
            
            # initialize network
            self.intermediate = tf.add(tf.matmul(self.s, self.weights[0]), self.biases[0]) # first layer with tensor t0 = x*W0+b0
            self.network = tf.add(tf.matmul(self.intermediate, self.weights[1]), self.biases[1]) #second layer with tensor y = t0*W1+b1
            
            # define losses
            #tf.losses.mean_squared_error(self.network, self.u)
            #loss = tf.get_collection(tf.GraphKeys.LOSSES)
            
            # define cost
            self.u_ = tf.nn.softmax(self.network)
            cost = tf.reduce_mean(-tf.reduce_sum(self.u_ * tf.log(self.u), reduction_indices=[1]))
            
            # define training step
            train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)
            
            # weights before training
            print(self.weights[1].eval())
            
            # train
            for i in range(1):
                batch = nnm.formatBatch(25)
                train_step.run(feed_dict={self.s: batch[0], self.u: batch[1]})
                
            # weights after training
            print(self.weights[1].eval())
            
            sess.close()
            
        # get a batch for our linear function test setup
        def linearFunctionBatch(self, size):
            x = []
            y = []
            for i in range(size):
                e_x = []
                e_y = []
                tmp = 0
                for j in range(4):
                    e_x.append(round(random.random()*10))
                    tmp += e_x[j]*0.25
                
                tmp += 1
                e_y.append(tmp)
                x.append(e_x)
                y.append(e_y)
            return [x, y]
            
        # In order to test all the tensor flow functionality and get
        # my head around the framework this will be a demonstration
        # of a simple linear system that has to average some input and 
        # add a bias with 4 random inputs thus: Y = 1/4*X + 1 = W*X + B
        # we will setup tensorflow such that it will find the values of W and B
        # by comparing its output to a function which already contains this function
        # it will thus try to learn that functions functionality
        def tensorFlowWithLinearSystem(self):
            # session
            sess = tf.Session()
            
            # tensor placeholders which will be loaded by training batches
            x = tf.placeholder(tf.float32, shape=[None, 4])
            y = tf.placeholder(tf.float32, shape=[None, 1])
            
            # weight and bias tensors 
            # network has 4 inputs and only 2 layers (input, output) with 1 neuron
            W = tf.Variable(tf.random_normal([4, 1]), tf.float32)
            b = tf.Variable(tf.random_normal([1]), tf.float32)
            
            model = tf.add(tf.matmul(x,W), b)
            
            sess.run(tf.global_variables_initializer())
            
            squared_delta = tf.square(model - y)
            loss = tf.reduce_sum(squared_delta)
            
            optimizer = tf.train.GradientDescentOptimizer(0.0005)
            train = optimizer.minimize(loss)

            # training
            for i in range(1000):
                #batch = self.linearFunctionBatch(1)
                #sess.run(train, {x: batch[0], y: batch[1]})
                sess.run(train, {x: [[1,2,3,4]], y: [[3.5]]})
                if i%100 == 0:
                    #print(sess.run(loss, {x: batch[0], y: [[3.5]]}))
                    print(sess.run(loss, {x: [[1,2,3,4]], y: [[3.5]]}))
                
            print(sess.run([W, b]))
            print(sess.run(model, {x : [[1,2,3,4]]}))
            
        # Training function
        def train(self):
            print("Starting training!")
            
            
            
