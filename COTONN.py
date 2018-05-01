from Importer import Importer
from NeuralNetworkManager import NeuralNetworkManager
from NeuralNetworkManager import NNTypes
# from NeuralNetworkManager import NNTrainingMethods

# Main class from which all functions are called
class COTONN:
    def __init__(self):
        self.importer = Importer()
        self.nn = NeuralNetworkManager()
        self.plainController = None
        
        self.debug_mode = False
        
        self.importer.setDebugMode(self.debug_mode)

    # Test function to automatically convert a plain controller to a simple MLP network
    def controllerToMLP(self, filename):
        self.plainController = self.importer.readPlainController(filename)
        self.nn.setNeuralNetworkType(NNTypes.MLP)
        

cotonn = COTONN()
cotonn.controllerToMLP("controllers/dcdc/controller")

if (cotonn.nn.nn_type == NNTypes.RBF):
    print("We are training a MLP topology network!")

