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
    
    # Formats a single state and input pair based on its index into a binary arrays representing the same data
    def formatSingleStateToBinaryPair(self, id):     
        pair = self.controller.getPairFromIndex(id)
        
        if(pair == None): return None
        
        pair[0] = self.bed.ntoba(pair[0], self.num_input_neurons)
        pair[1] = self.bed.ntoba(pair[1], self.num_output_neurons)
        
        return pair
    
    # Format a training batch from controller data for NN training for binary encoding of controller indices into a tensor
    def formatBinaryBatch(self, size):
        i = 0
        x = []
        y = []
        
        while i < size:
            s = round(random.random()*self.controller.size())
            p = self.formatSingleStateToBinaryPair(s)
            if(p != None):
                x.append(p[0])
                y.append(p[1])
                i += 1
                
        return [x, y]
        
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
    
    # Function which goes through all the states in the controller and checks if the neural network estimates the input correctly
    def checkFitness(self):
        size = self.controller.size()
        
        x = []
        y = []
        for i in range(size):
            p = self.formatSingleStateToBinaryPair(i)
            x.append(p[0])
            y.append(p[1])
            
        estimation = self.nn.estimate(x)
        fit = 0
        wrong = []
        
        for i in range(size):
            if(self.bed.etoba(estimation[i]) == y[i]):
                fit += 1
            else:
                wrong.append(i)
                
#        if fit < size:
#            print("Wrong for i = " + str(wrong[0]))
#            print("u_: " + str(self.bed.eton(estimation[wrong[0]])))
#            print("u:  " + str(self.bed.baton(y[wrong[0]])))
                
        return float(fit/size)
    
    # Reads the input from the neural net for a given state
    def readInputFromNN(self, s):
        est = self.nn.estimate([self.bed.ntoba(s, self.num_input_neurons)])[0]
        u = self.bed.eton(est)
        au = self.controller.getInputFromStateId(s)
        print("State: " + str(s) + " - Input: " + str(u) + " should be: " + str(au))
        return [s, u, au]
            
    def generateMLP(self, type, training_method, controller):
        self.type = type
        self.training_method = training_method
        self.controller = controller
        
        # initialize nn based on type
        if(self.type == NNTypes.MLP):
            self.nn = MLP()
            self.nn.setDebugMode(False)
            
        # initialize layers in nn    
        self.calculateIONeurons()
        self.nn.setNeurons(self.estimateNeurons(3))   
        
        print("Generated network neuron topology:")
        print(self.nn.getLayers())
        
        # initialize nn itself
        self.nn.initializeNetwork()
        self.nn.initializeLossFunction()
        
        #self.nn.setLearningRate(0.005)
        self.nn.setLearningRate(0.005)
        self.nn.setLossThreshold(1e-8)
        self.nn.setFitnessThreshold(0.99)
        self.nn.setBatchSize(200)
        self.nn.setDisplayStep(100)
        
        self.nn.initializeTrainFunction()
        
        # train nn
        self.nn.train(self)
        
        # checking fitness
        #print("\nChecking fitness:")
        #print("Fitness: " + str(float("{0:.3f}".format(self.checkFitness()*100))) + "%")
        
        # read value as check
        print("\nChecking NN:")
        self.readInputFromNN(502)
        
        # print some weights because we can
        #print(self.nn.session.run(self.nn.weights[0]))
        
        self.nn.close()
    
            

        
 

        