from Importer import Importer
from NeuralNetworkManager import NeuralNetworkManager
from BinaryEncoderDecoder import BinaryEncoderDecoder
from StaticController import StaticController
from DataSet import DataSet

from NeuralNetworkManager import NNTypes
from NeuralNetworkManager import NNOptimizer

# Main class from which all functions are called
class COTONN:
    def __init__(self):
        self.importer = Importer()
        self.nnm = NeuralNetworkManager()
        self.staticController = StaticController()
        self.dataSet = DataSet()
        
        self.debug_mode = True
        
        self.importer.setDebugMode(False)
        self.nnm.setDebugMode(self.debug_mode)

    # Test function to automatically convert a plain controller to a simple MLP network
    def run(self):      
        print("COTONN v0.2")
        
        # read static controller
        filename = "controllers/dcdc/controller" # for smaller network use simple
        self.staticController = self.importer.readStaticController(filename)
        
        # define dataset
        self.dataSet.readSetFromController(self.staticController)
        self.dataSet.formatToBinary()
        
        # specify neural network
        self.nnm.setType(NNTypes.MLP)
        self.nnm.setTrainingMethod(NNOptimizer.Adam)
        self.nnm.setDataSet(self.dataSet)
        self.nnm.initializeNeuralNetwork(1)
        
        # training
        self.nnm.initializeTraining(0.005, 0.975, 200, 1000, 1e4)
        self.nnm.train()
        
        # arbirary check
        print("\nChecking first value:")
        print(self.nnm.readSingle(self.dataSet.getX(0)))
        print(self.dataSet.getY(0))
        
        # save nn
        #self.nnm.save()
        
        # close session
        self.nnm.close()

cotonn = COTONN()
cotonn.run()

