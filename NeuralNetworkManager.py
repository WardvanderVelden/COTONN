# Class which will handle all the work done on neural networks and will contain all the functions which
# are called in tensorflow in order to generate neural networks from the controllers.

from MLP import MLP

class NeuralNetworkManager:
	def __init__(self):
		self.type = None
        
  
      """
   If we want to build a Multilayer Perceptron Neural Network:
         nn = MLP ()    we can use the MLP file for our type of network
         nn.BuildNN(data)  with the dataset we want to use for a network
   """
     
        
        