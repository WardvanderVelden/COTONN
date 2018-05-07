import tensorflow as tf
import NeuralNetworkManager
import random
import math

# The MLP class as presented in this file is a bare bone object oriented approach to creating an MLP
# for the purpose of capturing the behaviour of a static controller (table). 
class MLP:   
        def __init__(self):
            self.debug_mode = False
            
            # General parameters
            self.learning_rate = 0.1
            self.batch_size = 100
            self.display_step = 1000
            self.loss_threshold = 0.1
            self.fitness_threshold = 0.975
            
            # Network parameters
            self.layers = []
            self.num_layers = 0
            
            # Neural network general variables (MLP)
            self.s = None 
            self.u = None
            
            self.weights = []
            self.biases = []
            
            self.network = None 
            
            # Neural network training variables
            self.squared_delta = None
            self.loss_function = None
            self.train_function = None
            
            # Tensorflow specific
            self.session = tf.Session()
            
        # Setters
        def setLearningRate(self, value): self.learning_rate = value
        def setBatchSize(self, value): self.batch_size = value
        def setDisplayStep(self, value): self.display_step = value
        def setLossThreshold(self, value): self.loss_threshold = value
        def setFitnessThreshold(self, value): self.fitness_threshold = value
        def setDebugMode(self, value): self.debug_mode = value
      
        def setNeurons(self, layers):
            self.num_layers = len(layers)
            self.layers = layers
      
        # Getters
        def getLearningRate(self): return self.learning_rate
        def getBatchSize(self): return self.batch_size
        def getDisplayStep(self): return self.display_step
        def getLossThreshold(self): return self.loss_threshold
        def getFitnessThreshold(self): return self.fitness_threshold
        def getNumLayers(self): return self.num_layers
        def getLayers(self): return self.layers
        
        # Initialize network function which intializes an initial network with random weights and biases
        def initializeNetwork(self):
            # Initialize tensors
            self.x = tf.placeholder(tf.float32, [None, self.layers[0]])
            self.y = tf.placeholder(tf.float32, [None, self.layers[-1]])
            
            self.weights = [None]*(self.num_layers - 1)
            self.biases = [None]*(self.num_layers - 1)
            
            for i in range(self.num_layers - 1):
                self.weights[i] = tf.Variable(tf.random_normal([self.layers[i], self.layers[i+1]]),tf.float32)
                self.biases[i]= tf.Variable(tf.random_normal([self.layers[i+1]]),tf.float32)
                

            # Define network
            tf_layers = [None]*(self.num_layers - 1)   
            tf_layers[0] = tf.sigmoid(tf.add(tf.matmul(self.x, self.weights[0]), self.biases[0])) # input layer
            for i in range(1, self.num_layers - 1):
                tf_layers[i] = tf.sigmoid(tf.add(tf.matmul(tf_layers[i-1], self.weights[i]), self.biases[i]))
                
            self.network = tf_layers[-1]
            
            self.session.run(tf.global_variables_initializer())
            
            return self.network
        
        # Initialize loss function
        def initializeLossFunction(self):
            self.squared_delta = tf.square(self.network - self.y)
            self.loss_function = tf.reduce_sum(self.squared_delta)
            
            return self.loss_function
        
        # Intialize training function
        def initializeTrainFunction(self):
            #self.train_function = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss_function)
            self.train_function = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_function)
            #self.train_function = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss_function)
            #self.train_function = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(self.loss_function)
            
            self.session.run(tf.global_variables_initializer())
            
            return self.train_function
            
        # Training function
        def train(self, nnm):
            i = 0
            l = 0
            fit = 0.0
            
            print("\nTraining network:")
            
            while True:
                # get mini batch
                mini_batch = nnm.formatBinaryBatch(self.batch_size)
                
                # do one gradient descent step using the random mini batch
                self.session.run(self.train_function, {self.x: mini_batch[0], self.y: mini_batch[1]})
                
                # determine the loss given the correct y value
                l = self.session.run(self.loss_function, {self.x: mini_batch[0], self.y: mini_batch[1]})
                
                if i % self.display_step == 0 and i != 0:
                    fit = nnm.checkFitness()
                    print("i = " + str(i) + "\tloss = " + str(float("{0:.5f}".format(l))) + "\t\tfit = " + str(float("{0:.5f}".format(fit))))
                    if(self.debug_mode):
                        est = nnm.bed.etoba(self.session.run(self.network, {self.x: [mini_batch[0][0]]})[0])
                        print("u:  " + str(mini_batch[1][0]))
                        print("u_: " + str(est))
                
                if l <= self.loss_threshold or fit >= self.fitness_threshold:
                    print("Done training at:")
                    print("i = " + str(i) + "\tloss = " + str(float("{0:.5f}".format(l))) + "\t\tfit = " + str(float("{0:.5f}".format(fit))))
                    break
                
                if math.isnan(l):
                    print("Loss diverged at i = " + str(i))
                    break
                
                i += 1
                
        # Estimator function which estimates the desired outcome based on an input
        def estimate(self, x):
            return self.session.run(self.network, {self.x: x})
            
            
        # Get a batch for our linear function test setup
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
                
                tmp += 2.5
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
            
            loss_threshold = 1e-8
            
            # training
            train = tf.train.GradientDescentOptimizer(0.000075).minimize(loss)
            
            i = 0
            l = 0
            old_l = l
            
            while True:
                # get mini batch
                mini_batch = self.linearFunctionBatch(100)
                
                # do one gradient descent step using the random mini batch
                sess.run(train, {x: mini_batch[0], y: mini_batch[1]})
                
                # determine the loss given the correct y value
                l = sess.run(loss, {x: mini_batch[0], y: mini_batch[1]})
                
                if i % self.display_step == 0 and i != 0:
                    print("i = " + str(i) + " \t loss = " + str(l))
                
                if l < loss_threshold:
                    print("Done training at i = " + str(i) + " with loss = " + str(l))
                    break
                
                if old_l == l:
                    print("Training no longer improves loss at i = " + str(i) + " with loss = " + str(l))
                    break
                
                if math.isnan(l):
                    print("Loss diverged at i = " + str(i))
                    break
                
                i += 1
                old_l = l
                
            eval_W = sess.run(W)
            eval_b = sess.run(b)
            print("W: " + str(eval_W))
            print("b: " + str(eval_b))
            print(sess.run(model, {x : [[1,2,3,4]]}))
            
            sess.close()
            
        # Close function which closes tensorflow session
        def close(self):
            self.session.close()
            
            
            
