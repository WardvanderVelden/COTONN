# Class which will handle all the work done on neural networks and will contain all the functions which
# are called in tensorflow in order to generate neural networks from the controllers.

from MLP import MLP
from BinaryEncoderDecoder import BinaryEncoderDecoder
from enum import Enum

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
        bed = BinaryEncoderDecoder()
        
        h_state = self.controller.getHighestStateID()
        h_input = self.controller.getHighestInputID()
        
        b_state = bed.sntob(h_state)
        b_input = bed.sntob(h_input)
        
        self.num_input_neurons = len(b_state)
        self.num_output_neurons = len(b_input)
        
    # Estimate the hidden layer neurons based on input and output neurons and amount of hidden layers
    def estimateHiddenNeurons(self, num_hidden_layers):
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
            
    def initialize(self, type, training_method, controller):
        self.type = type
        self.training_method = training_method
        self.controller = controller
        
        # initialize nn based on type
        if(self.type == NNTypes.MLP):
            self.nn = MLP()
            
        # initialize layers in nn    
        self.calculateIONeurons()
        self.nn.setNeurons(self.estimateHiddenNeurons(3))   
        
        # initialize nn in itself
        self.nn.initialize()
        
        
    def train(self):
        if(self.nn != None):
            self.nn.train()
        
            

        
 

        