from Importer import Importer
from NeuralNetworkManager import NeuralNetworkManager
from BinaryEncoderDecoder import BinaryEncoderDecoder
from StaticController import StaticController

from NeuralNetworkManager import NNTypes
from NeuralNetworkManager import NNOptimizer

# Main class from which all functions are called
class COTONN:
    def __init__(self):
        self.importer = Importer()
        self.nnm = NeuralNetworkManager()
        self.bed = BinaryEncoderDecoder()
        self.staticController = StaticController()
        
        self.debug_mode = True
        
        self.importer.setDebugMode(False) # Disable debug mode for the importer to prevent intermediate import results
        self.nnm.setDebugMode(self.debug_mode)

    # Test function to automatically convert a plain controller to a simple MLP network
    def test(self):
        # read static controller
        filename = "controllers/dcdc/controller" # for smaller network use simple
        self.staticController = self.importer.readStaticController(filename)
        
        # initialize neural network
        self.nnm.generateMLP(NNTypes.MLP, NNOptimizer.Gradient_Descent, self.staticController)

cotonn = COTONN()
cotonn.test()

