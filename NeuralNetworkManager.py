# Class which will handle all the work done on neural networks and will contain all the functions which
# are called in tensorflow in order to generate neural networks from the controllers.

from MLP import MLP
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
        self.nn_type = None
        self.nn_training_method = None
        self.nn = None
        
    # Set neural network type to desired type
    def setNeuralNetworkType(self, nn_type):
        self.nn_type = nn_type
        
    # Set neural network training method to the desired method
    def setTrainingMethod(self, nn_method):
        set.nn_training_method = nn_method
 

        