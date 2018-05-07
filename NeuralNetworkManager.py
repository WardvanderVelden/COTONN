#import tensorflow as tf
from BinaryEncoderDecoder import BinaryEncoderDecoder
from MLP import MLP
from enum import Enum
import math

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
      Linear = 3

# Class which will handle all the work done on neural networks and will contain all the functions which
# are called in tensorflow in order to generate neural networks from the controllers.
class NeuralNetworkManager:
    def __init__(self):
        self.type = None
        self.training_method = None
        self.activation_function = None
        
        self.learning_rate = 0.1
        self.fitness_threshold = 0.75
        self.batch_size = 100
        self.display_step = 1000
        
        self.epoch = 0
        self.epoch_threshold = 10000
        
        self.layers = []
        
        self.data_set = None
        
        self.bed = BinaryEncoderDecoder()
        
        self.debug_mode = False
        
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
    
    def setType(self, type): self.type = type
    def setTrainingMethod(self, optimizer): self.training_method = optimizer
    def setActivationFunction(self, activation_function): self.activation_function = activation_function
    def setLearningRate(self, value): self.learning_rate = value
    def setFitnessThreshold(self, value): self.fitness_threshold = value
    def setBatchSize(self, value): self.batch_size = value
    def setDisplayStep(self, value): self.display_step = value
    def setEpochThreshold(self, value): self.epoch_threshold = value

    def setDataSet(self, data_set): self.data_set = data_set
    
    def setDebugMode(self, value): self.debug_mode = value
    
    # Reads the input from the neural net for a given state
    def readSingle(self, x):
        return self.nn.estimate([x])[0]
        
    # Estimate the hidden layer neurons based on input and output neurons and amount of hidden layers
    def estimateNeurons(self, num_hidden_layers):
        self.layers = []
        
        x_dim = self.data_set.getXDim()
        y_dim = self.data_set.getYDim()
        
        a = (y_dim - x_dim)/(num_hidden_layers + 1)
        
        self.layers.append(x_dim)
        for i in range(1, num_hidden_layers + 1):
            self.layers.append(round(x_dim + a*i))
        self.layers.append(y_dim)
        
        return self.layers
    
    # Function which goes through all the states in the controller and checks if the neural network estimates the input correctly
    def checkFitness(self):
        size = self.data_set.getSize()
            
        estimation = self.nn.estimate(self.data_set.x)
        fit = 0
        wrong = []
        
        for i in range(size):
            if(self.bed.etoba(estimation[i]) == self.data_set.y[i]):
                fit += 1
            else:
                wrong.append(i)
                
        return float(fit/size)
    
    # Initialize neural network
    def initializeNeuralNetwork(self, hidden):
        if(self.type == NNTypes.MLP):
            self.nn = MLP()
            self.nn.setDebugMode(False)
            
        # Initialize network and loss function
        self.estimateNeurons(hidden)
        self.nn.setNeurons(self.layers)
        self.nn.initializeNetwork(self.activation_function)
        
        print("\nGenerated network neuron topology:")
        print(self.layers)
        
        
    def initializeTraining(self, learning_rate, fitness_threshold, batch_size, display_step, epoch_threshold = 1e6):
        self.learning_rate = learning_rate
        self.fitness_threshold = fitness_threshold
        
        self.batch_size = batch_size
        self.display_step = display_step
        
        self.epoch_threshold = epoch_threshold
        
        self.nn.initializeLossFunction()
        self.nn.initializeTrainFunction(self.training_method, self.learning_rate)
        
        print("\nTraining setting initialized")
        
        
    def train(self):
        print("\nTraining:")
        i = 0
        batch_index = 0
        loss = 0
        old_loss = loss
        fit = 0.0
        
        while True:
            batch = self.data_set.getBatch(self.batch_size, batch_index)
            loss = self.nn.trainStep(batch)
            
            if(i % self.display_step == 0 and i != 0):
                fit = self.checkFitness()
                print("i = " + str(i) + "\tepoch = " + str(self.epoch) + "\tloss = " + str(float("{0:.3f}".format(loss))) + "\tfit = " + str(float("{0:.3f}".format(fit))))
                
            if(self.epoch > self.epoch_threshold or fit >= self.fitness_threshold):
                print("Finished training")
                break
            
            if(math.isnan(loss) or old_loss == loss):
                print("i = " + str(i) + "\tepoch = " + str(self.epoch) + "\tloss = " + str(float("{0:.3f}".format(loss))) + "\tfit = " + str(float("{0:.3f}".format(fit))))
                print("Finished training, solution did not converge")
                break
            
            batch_index += self.batch_size
            if(batch_index >= self.data_set.getSize()): 
                batch_index = batch_index % self.data_set.getSize()
                self.epoch += 1
            i += 1
            old_loss = loss
            
        
    def save(self):
        print("\nSaving neural network")
    
    def close(self):
        self.nn.close()
        
            

        
 

        