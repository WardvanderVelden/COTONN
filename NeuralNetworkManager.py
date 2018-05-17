#import tensorflow as tf
from BinaryEncoderDecoder import BinaryEncoderDecoder
from MLP import MLP
from enum import Enum

import tensorflow as tf
import math
import numpy
import signal
import time
from time import gmtime, strftime
import matplotlib.pyplot as plt
import random


class NNTypes(Enum):
      MLP = 1
      RBF = 2
      CMLP = 3
    
    
class NNOptimizer(Enum):
      Gradient_Descent = 1
      Adagrad = 2
      Adadelta = 3
      Adam = 4
      Ftrl = 5
      Momentum = 6
      RMSProp =7    
    
    
class NNActivationFunction(Enum):
      Sigmoid = 1
      Relu = 2
      tanh = 3

      
# Class which will handle all the work done on neural networks and will contain all the functions which
# are called in tensorflow in order to generate neural networks from the controllers.
class NeuralNetworkManager:
    def __init__(self):
        self.type = None
        self.nn = MLP()
        self.training_method = None
        self.activation_function = None
        self.dropout_rate = 0.0
        
        self.training = True
        self.learning_rate = 0.1
        self.fitness_threshold = 0.75
        self.batch_size = 100
        self.shuffle_rate = 2500
        self.display_step = 1000
        
        self.epoch = 0
        
        self.layers = []
        
        self.data_set = None
        
        self.bed = BinaryEncoderDecoder()
        
        self.debug_mode = False
        
        # Plotting variables
        self.losses = []
        self.fitnesses = []
        self.iterations = []
        
        self.tensorboard_log_path = './nn/log/'
        

    # Getters and setters
    def getType(self): return self.type
    def getTrainingMethod(self): return self.training_method
    def getActivationFunction(self): return self.activation_function
    def getLearningRate(self): return self.learning_rate
    def getFitnessThreshold(self): return self.fitness_threshold
    def getBatchSize(self): return self.batch_size
    def getDisplayStep(self): return self.display_step
    def getEpoch(self): return self.epoch
    def getEpochThreshold(self): return self.epoch_threshold
    def getDropoutRate(self): return self.dropout_rate
    def getShuffleRate(self): return self.shuffle_rate
    def getSaveLocation(self): return self.save_location
    
    def setType(self, type): self.type = type
    def setTrainingMethod(self, optimizer): self.training_method = optimizer
    def setActivationFunction(self, activation_function): self.activation_function = activation_function
    def setLearningRate(self, value): self.learning_rate = value
    def setFitnessThreshold(self, value): self.fitness_threshold = value
    def setBatchSize(self, value): self.batch_size = value
    def setDisplayStep(self, value): self.display_step = value
    def setEpochThreshold(self, keep_probability): self.epoch_threshold = keep_probability
    def setDropoutRate(self, value): self.dropout_rate = value
    def setShuffleRate(self, value): self.shuffle_rate = value
    def setSaveLocation(self, value): self.save_location = value
    def setDebugMode(self, value): self.debug_mode = value
    def setDataSet(self, data_set): self.data_set = data_set
    
    
    # Hidden layer generation functions
    # Linearly increase/decrease neurons per hidden layer based on the input and ouput neurons
    def linearHiddenLayers(self, num_hidden_layers):
        self.layers = []
        
        x_dim = self.data_set.getXDim()
        y_dim = self.data_set.getYDim()
        
        a = (y_dim - x_dim)/(num_hidden_layers + 1)
        
        self.layers.append(x_dim)
        for i in range(1, num_hidden_layers + 1):
            self.layers.append(round(x_dim + a*i))
        self.layers.append(y_dim)
        
        return self.layers
    
    
    # Rectangular hidden layer
    def rectangularHiddenLayers(self, width, height):
        self.layers = []
        
        self.layers.append(self.data_set.getXDim())
        for i in range(width):
            self.layers.append(height)
        self.layers.append(self.data_set.getYDim())


     # Initialize neural network
    def initializeNeuralNetwork(self):
        if(self.debug_mode):
            print("\nNeural network initialization:")

        if(self.type == NNTypes.MLP):
            self.nn = MLP()
            self.nn.setDebugMode(self.debug_mode)
            if(self.debug_mode):
                print("Neural network type: MLP")
            
        # Initialize network and loss function
        self.nn.setNeurons(self.layers)
        self.nn.setDropoutRate(self.dropout_rate)
        self.nn.setActivationFunction(self.activation_function)
        self.nn.initializeNetwork()
        
        # Print neural network status
        if(self.debug_mode):
            print("Generated network neuron topology: " + str(self.layers) + " with dropout rate: " + str(self.nn.getDropoutRate()))
        
        
    # Initialize training function
    def initializeTraining(self, learning_rate, fitness_threshold, batch_size, display_step, epoch_threshold = -1, shuffle_rate = 10000):      
        self.learning_rate = learning_rate
        self.fitness_threshold = fitness_threshold
        self.batch_size = batch_size
        self.display_step = display_step
        self.epoch_threshold = epoch_threshold
        self.shuffle_rate = shuffle_rate
        
        self.nn.initializeLossFunction()
        self.nn.initializeTrainFunction(self.training_method, self.learning_rate)
        
        
    # Initialize fitness function
    def initializeFitnessFunction(self):
        with tf.name_scope("fitness"):
            eta = self.data_set.getYEta()
            size = self.data_set.getSize()
            
            lower_bound = tf.subtract(self.nn.y, eta)
            upper_bound = tf.add(self.nn.y, eta)
            
            is_fit = tf.logical_and(tf.greater_equal(self.nn.predictor, lower_bound), tf.less(self.nn.predictor, upper_bound))
            non_zero = tf.to_float(tf.count_nonzero(tf.reduce_min(tf.cast(is_fit, tf.int8), 1)))
            self.fitness = non_zero/size
            
            tf.summary.scalar("fitness", self.fitness)
        
        
    # General initialization function to call all functions
    def initialize(self, learning_rate, fitness_threshold, batch_size, display_step, epoch_threshold = -1, shuffle_rate = 10000):
        self.initializeNeuralNetwork()
        self.initializeFitnessFunction()
        self.initializeTraining(learning_rate, fitness_threshold, batch_size, display_step, epoch_threshold, shuffle_rate)
        
        time_stamp = strftime("%y%m%d%H%M%S", gmtime())
        self.train_writer = tf.summary.FileWriter(self.tensorboard_log_path + time_stamp, self.nn.session.graph)
        
        
    # Check a state against the dataset and nn by using its id in the dataset
    def checkByIndex(self, index, out):
        x = self.data_set.x[index]
        estimation = self.nn.estimate([x])[0]
        y = self.data_set.getY(index)
        
        y_eta = self.data_set.getYEta()
        equal = True
        for i in range(self.data_set.getYDim()):
            if(not((y[i] - y_eta[i]) <= estimation[i] and (y[i] + y_eta[i]) > estimation[i])):
                equal = False
        
        if(out):
            print("u: " + str(y) + " u_: " + str(numpy.round(estimation,2)) + " within etas: " + str(equal))
            
        return equal
    
    
    # Check fitness of the neural network for a specific dataset and return wrong states
    # as of right now it assumes a binary encoding of the dataset
    def checkFitness(self, data_set):
        self.data_set = data_set
        
        size = self.data_set.getSize()
        fit = size
        
        wrong = []
        
        x, y = self.data_set.x, self.data_set.y
        y_eta = self.data_set.getYEta()
        y_dim = self.data_set.getYDim()
        
        estimation = self.nn.estimate(self.data_set.x)
        
        for i in range(size):
            equal = True
            for j in range(y_dim):
                if(not((y[i][j] - y_eta[j]) <= estimation[i][j] and (y[i][j] + y_eta[j]) > estimation[i][j]) and equal):
                    wrong.append(self.bed.baton(x[i]))
                    fit -= 1
                    equal = False
                    
        fitness = fit/size*100
        print("\nDataset fitness: " + str(float("{0:.3f}".format(fitness))) + "%")
                    
        return fitness, wrong


    # Randomly check neural network against a dataset
    def randomCheck(self, data_set):
        self.data_set = data_set
        
        self.initializeFitnessFunction()        

        print("\nValidating:")
        for i in range(10):
            r = round(random.random()*(self.data_set.getSize()-1))
            self.checkByIndex(r, True)

        
    # Train network
    def train(self):
        self.clear()
        
        print("\nTraining (Ctrl+C to interrupt):")
        signal.signal(signal.SIGINT, self.interrupt)

        i, batch_index, loss, fit = 0,0,0,0.0

        self.merged_summary = tf.summary.merge_all()
        
        start_time = time.time()
        while self.training:
            batch = self.data_set.getBatch(self.batch_size, batch_index)
            loss, summary = self.nn.trainStep(batch, self.merged_summary)
            
            if(i % self.shuffle_rate == 0 and i != 0): self.data_set.shuffle()
            
            if(i % self.display_step == 0 and i != 0):
                fit = self.nn.runInSession(self.fitness, self.data_set.x, self.data_set.y)
                
                self.addToLog(loss, fit, i)
                print("i = " + str(i) + "\tepoch = " + str(self.epoch) + "\tloss = " + str(float("{0:.3f}".format(loss))) + "\tfit = " + str(float("{0:.3f}".format(fit))))
                self.train_writer.add_summary(summary, i)
                
            if(self.epoch >= self.epoch_threshold and self.epoch_threshold > 0):
                print("i = " + str(i) + "\tepoch = " + str(self.epoch) + "\tloss = " + str(float("{0:.3f}".format(loss))) + "\tfit = " + str(float("{0:.3f}".format(fit))))
                print("Finished training, epoch threshold reached")
                break
            
            if(fit >= self.fitness_threshold):
                print("Finished training")
                break
            
            if(math.isnan(loss)):
                print("i = " + str(i) + "\tepoch = " + str(self.epoch) + "\tloss = " + str(float("{0:.3f}".format(loss))) + "\tfit = " + str(float("{0:.3f}".format(fit))))
                print("Finished training, solution did not converge")
                break
            
            batch_index += self.batch_size
            if(batch_index >= self.data_set.getSize()): 
                batch_index = batch_index % self.data_set.getSize()
                self.epoch += 1
            
            i += 1
        

        self.weights_layer = []
        self.num_layers = self.nn.getNumLayers()
        # Save variables from layers
        for i in range(self.num_layers -1):
              with tf.variable_scope("Layer_"+str(i+1), reuse=True):
                    self.weights_layer.append(tf.get_variable("kernel"))
 
        end_time = time.time()
        print("Time taken: " + self.formatTime(end_time - start_time))
        
        
    # Format
    def formatTime(self, time):
        h = math.floor(time / 3600)
        m = math.floor(time / 60) % 60
        s = time - h*3600 - m*60
        
        return str(h)+" hrs "+str(m)+" mins "+str(float("{0:.2f}".format(s)))+" secs"
            
        
    # Interrupt handler to interrupt the training while in progress
    def interrupt(self, signal, frame):
        self.training = False
          
        
    # Plotting loss and fitness functions
    def plot(self):      
        plt.figure(1)
        plt.plot(self.iterations, self.losses, 'bo')
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.grid()
        x1,x2,y1,y2 = plt.axis()
        plt.axis((x1,x2,0,y2+0.1))
        
        plt.figure(2)
        plt.plot(self.iterations, self.fitnesses, 'r-')
        plt.xlabel("Iterations")
        plt.ylabel("Fitness")
        plt.grid()
        x1,x2,y1,y2 = plt.axis()
        plt.axis((x1,x2,0,1))
        plt.show()
        

    # Add to log
    def addToLog(self, loss, fit, iteration):
        self.losses.append(loss)
        self.fitnesses.append(fit)
        self.iterations.append(iteration)

    # Clear variables
    def clear(self):
        self.epoch = 0
        self.training = True

        self.fitnesses = []
        self.iterations = []
        self.losses = []
        
    # Save network
    def save(self, filename):
        print("\nSaving neural network")
        self.nn.save(filename)
    

    # Close session
    def close(self):
        self.nn.close()
        self.train_writer.close()
        
