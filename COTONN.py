from Importer import Importer
from NeuralNetworkManager import NeuralNetworkManager
from StaticController import StaticController
from DataSet import DataSet

import random

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
        filename = "controllers/dcdc/simple" # for smaller network use simple
        self.staticController = self.importer.readStaticController(filename)
        
        # define dataset
        self.dataSet.readSetFromController(self.staticController)
        self.dataSet.formatToBinary()
        
        # specify neural network
        self.nnm.setType(NNTypes.MLP)
        self.nnm.setTrainingMethod(NNOptimizer.Adam)
        self.nnm.setDataSet(self.dataSet)
        self.nnm.rectangularHiddenLayers(3, 8)
        self.nnm.initializeNeuralNetwork()
        
        # training
        self.nnm.initializeTraining(0.01, 1.0, 25, 1000)
        self.nnm.train()
        
        # validate by randomly picking inputs
        print("\nValidating:")
        for i in range(10):
            r = round(random.random()*(self.dataSet.getSize()-1))
            self.nnm.checkByIndex(r)
        
        # save nn
        #self.nnm.save()
        
        # close session
        self.nnm.close()

cotonn = COTONN()
cotonn.run()

