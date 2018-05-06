# Class which will handle all the work done on neural networks and will contain all the functions which
# are called in tensorflow in order to generate neural networks from the controllers.

import tensorflow as tf
from MLP import MLP
from BinaryEncoderDecoder import BinaryEncoderDecoder
from enum import Enum
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
      Linear = 3

class NeuralNetworkManager:
    def __init__(self):
        self.type = None
        self.training_method = None
        
        self.controller = None;
        
        self.bed = BinaryEncoderDecoder()
        
        self.num_input_neurons = 0
        self.num_output_neurons = 0
        
        self.debug_mode = False
        
    # Getters and setters
    def getType(self): return self.type
    def getTrainingMethod(self): return self.training_method
    
    def setType(self, type): self.type = type
    def setTrainingMethod(self, method): self.training_method = method
    def setController(self, controller): self.controller = controller
    
    def setDebugMode(self, value): self.debug_mode = value
        
    # Calculate the required amount of neural needed for both the input and the output based on the amount of 
    # binary numbers required to fully describe the state space and input space
    def calculateIONeurons(self):
        h_state = self.controller.getHighestStateID()
        h_input = self.controller.getHighestInputID()
        
        b_state = self.bed.sntob(h_state)
        b_input = self.bed.sntob(h_input)
        
        self.num_input_neurons = len(b_state)
        self.num_output_neurons = len(b_input)
        
    # Estimate the hidden layer neurons based on input and output neurons and amount of hidden layers
    def estimateNeurons(self, num_hidden_layers):
        layers = [0]*(num_hidden_layers+2)
        
        a = (self.num_output_neurons - self.num_input_neurons)/(num_hidden_layers + 1)
        
        # input layer
        layers[0] = self.num_input_neurons
        
        # hidden layers
        for i in range(1, num_hidden_layers + 1):
            layers[i] = round(self.num_input_neurons + a*i)
            
        # output layer
        layers[num_hidden_layers + 1] = self.num_output_neurons
        
        return layers
    
    # Format a single state into a binary encoded pair
    def formatSingleState(self, id):     
        pair = self.controller.getPairFromIndex(id)
        
        if(pair == None): return None
        
        pair[0] = self.bed.ntob(pair[0], self.num_input_neurons)
        pair[1] = self.bed.ntob(pair[1], self.num_output_neurons)
        
        return pair
    
    # Format a training batch from controller data for NN training for binary encoding of controller indices into a tensor
    def formatBatch(self, size):
        pairs = []
        
        while len(pairs) < size:
            s = round(random.random()*self.controller.size())
            p = self.formatSingleState(s)
            if(p != None):
                pairs.append(p)
                
        x = []
        y = []
        for i in range(size):
            p = pairs[i]
            e_x = []
            e_y = []
            for j in range(self.num_input_neurons):
                e_x.append(int(p[0][j]))
            for j in range(self.num_output_neurons):
                e_y.append(int(p[1][j]))
            
            x.append(e_x)
            y.append(e_y)
        
        return [x, y]
    
    # Function which goes through all the states in the controller and checks if the neural network estimates the input correctly
    def checkFitness(self):
        size = self.controller.size()
        
        x = []
        y = []
        for i in range(size):
            p = self.formatSingleState(i)
            e_x = []
            e_y = []
            for j in range(self.num_input_neurons):
                e_x.append(int(p[0][j]))
            for j in range(self.num_output_neurons):
                e_y.append(int(p[1][j]))
                
            x.append(e_x)
            y.append(e_y)
            
        estimation = self.nn.estimate(x)
        fit = 0
        
        for i in range(size):
            equal = True
            for j in range(self.num_output_neurons):
                if(round(estimation[i][j]) != float(y[i][j])):
                    equal = False
                    
            if(equal):
                fit += 1
                
        return float("{0:.5f}".format(fit/size))
            
    def generateMLP(self, type, training_method, controller):
        self.type = type
        self.training_method = training_method
        self.controller = controller
        
        # initialize nn based on type
        if(self.type == NNTypes.MLP):
            self.nn = MLP()
            
        # initialize layers in nn    
        self.calculateIONeurons()
        self.nn.setNeurons(self.estimateNeurons(3))   
        
        print("Generated network neuron topology:")
        print(self.nn.getLayers())
        
        # initialize nn itself
        self.nn.initializeNetwork()
        self.nn.initializeLossFunction()
        
        self.nn.setLearningRate(0.15)
        self.nn.setLossThreshold(1e-2)
        self.nn.setBatchSize(100)
        self.nn.setDisplayStep(1000)
        
        self.nn.initializeTrainFunction()
        
        # train nn
        self.nn.train(self)
        
        # checking fitness
        print("Checking fitness:")
        print("Fitness: " + str(self.checkFitness()*100) + "%")
        
        # printing some weights because we can
        #print(self.nn.session.run(self.nn.weights[0]))
        
        self.nn.close()
    
            

        
 

        