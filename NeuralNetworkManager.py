#import tensorflow as tf
from BinaryEncoderDecoder import BinaryEncoderDecoder
from MLP import MLP
from enum import Enum

import tensorflow as tf
import math
import numpy
import signal
import time
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
        self.training_method = None
        self.activation_function = None
        self.keep_prob = 1.0
        
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
        tf.summary.scalar('Loss', self.losses)
        self.fitnesses = []
        self.iterations = []
        
        self.tensorboard_log_path = './tmp/log'
        "--> tensorboard --logdir=log "
        
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
    def getKeepProbability(self): return self.keep_prob
    def getShuffleRate(self): return self.shuffle_rate
    
    def setType(self, type): self.type = type
    def setTrainingMethod(self, optimizer): self.training_method = optimizer
    def setActivationFunction(self, activation_function): self.activation_function = activation_function
    def setLearningRate(self, value): self.learning_rate = value
    def setFitnessThreshold(self, value): self.fitness_threshold = value
    def setBatchSize(self, value): self.batch_size = value
    def setDisplayStep(self, value): self.display_step = value
    def setEpochThreshold(self, keep_probability): self.epoch_threshold = keep_probability
    def setKeepProbability(self, value): self.keep_prob = value
    def setShuffleRate(self, value): self.shuffle_rate = value

    def setDataSet(self, data_set): self.data_set = data_set
    
    def setDebugMode(self, value): self.debug_mode = value
    
    
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
    def initializeNeuralNetwork(self, keep_prob = 1):
        if(self.debug_mode):
            print("\nNeural network initialization:")

        if(self.type == NNTypes.MLP):
            self.nn = MLP()
            self.nn.setDebugMode(self.debug_mode)
            if(self.debug_mode):
                print("Neural network type: MLP")
            
        # Initialize network and loss function
        self.nn.setNeurons(self.layers)
        self.nn.setKeepProbability(self.keep_prob)
        self.nn.setActivationFunction(self.activation_function)
        self.nn.initializeNetwork(self)
        
        # Print neural network status
        if(self.debug_mode):
            print("Generated network neuron topology: " + str(self.layers) + " with keep probability: " + str(self.nn.getKeepProbability()))
        
        
    # Initialize training function
    def initializeTraining(self, learning_rate, fitness_threshold, batch_size, display_step, shuffle_rate = 500, epoch_threshold = -1):
        if(self.debug_mode):
            print("\nInitializing training:")
            
        print("Generated network neuron topology: " + str(self.layers))
        if(self.keep_prob != 1.0):
            print("Neuron keep probability: " + str(self.nn.getKeepProbability()))
        tf.summary.scalar('keep_prob', self.keep_prob)
            
        self.learning_rate = learning_rate
        self.fitness_threshold = fitness_threshold
        self.batch_size = batch_size
        self.display_step = display_step
        self.epoch_threshold = epoch_threshold
        self.shuffle_rate = shuffle_rate
        
        self.nn.initializeLossFunction()
        self.nn.initializeTrainFunction(self.training_method, self.learning_rate)
        
        # Tensorboard
        self.train_writer = tf.summary.FileWriter(self.tensorboard_log_path, self.nn.session.graph)
        self.merged_summary = tf.summary.merge_all()
        return self.train_writer
        
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


    # Randomly check neural network against a dataset
    def randomCheck(self, data_set):
        self.data_set = data_set
        fit = self.checkFitness()
        print("\nDataset fitness: " + str(float("{0:.3f}".format(fit))))
        
        print("\nValidating:")
        for i in range(10):
            r = round(random.random()*(self.data_set.getSize()-1))
            self.checkByIndex(r, True)
    
    
    # Calculate fitness based on the dataset
    def checkFitness(self):
        size = self.data_set.getSize()
        
        with tf.name_scope('Estimation'):
              estimation = self.nn.estimate(self.data_set.x)
        fit = 0
        wrong = []
        
        y_eta = self.data_set.getYEta()
        y_dim = self.data_set.getYDim()
        
        for i in range(size):
            y = self.data_set.getY(i)
            equal = True
            for j in range(y_dim):
                if(not((y[j] - y_eta[j]) <= estimation[i][j] and (y[j] + y_eta[j]) > estimation[i][j])):
                    equal = False
                
            if(equal):
                fit += 1
            else:
                wrong.append(i)
        
        summary_fitness = float(fit/(size + self.epoch*self.data_set.getSize()))
        tf.summary.scalar('fitness', summary_fitness) 
        self.merged_summary = tf.summary.merge_all()    
        
        return float(fit/size), self.merged_summary

        
    # Train network
    def train(self):
        #self.clear()
        
        print("\nTraining (Ctrl+C to interrupt):")
        signal.signal(signal.SIGINT, self.interrupt)

        i, batch_index, loss, fit = 0,0,0,0.0
        self.merged_summary = tf.summary.merge_all()
        
        start_time = time.time()
        while self.training:
            batch = self.data_set.getBatch(self.batch_size, batch_index)
            loss, summary_result = self.nn.trainStep(batch, self.merged_summary)
            
            if(i % self.shuffle_rate == 0 and i != 0): self.data_set.shuffle()
            
            if(i % self.display_step == 0 and i != 0):
                fit, fitness_summary = self.checkFitness()
                self.addToLog(loss, fit, i)
                print("i = " + str(i) + "\tepoch = " + str(self.epoch) + "\tloss = " + str(float("{0:.3f}".format(loss))) + "\tfit = " + str(float("{0:.3f}".format(fit))))
                self.merged_summary = tf.summary.merge_all()
                
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
            #self.merged_summary = tf.summary.merge_all()
            self.train_writer.add_summary(summary_result, i)
            i += 1
        
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
        
            

        
 

        