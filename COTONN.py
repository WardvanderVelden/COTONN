from Importer import Importer
from Exporter import Exporter
from NeuralNetworkManager import NeuralNetworkManager
from StaticController import StaticController
from DataSet import DataSet

import random

from NeuralNetworkManager import NNTypes
from NeuralNetworkManager import NNOptimizer
from NeuralNetworkManager import NNActivationFunction

# Main class from which all functions are called
class COTONN:
    def __init__(self):
        self.importer = Importer()
        self.exporter = Exporter()
        self.nnm = NeuralNetworkManager()
        self.staticController = StaticController()
        self.dataSet = DataSet()
        
        self.debug_mode = True
        
        self.importer.setDebugMode(False)
        self.nnm.setDebugMode(self.debug_mode)

    # Test function to automatically convert a plain controller to a simple MLP network
    def run(self):      
        print("COTONN v0.2.3")
        
        # read static controller
        filename = "controllers/dcdc/simple" # for smaller network use simple
        self.staticController = self.importer.readStaticController(filename)
        
        # define dataset
        self.dataSet.readSetFromController(self.staticController)
        self.dataSet.formatToBinary()
        
        # specify neural network
        self.nnm.setType(NNTypes.MLP)
        self.nnm.setTrainingMethod(NNOptimizer.Adam)
        self.nnm.setActivationFunction(NNActivationFunction.Sigmoid)
        self.nnm.setDataSet(self.dataSet)
        self.nnm.rectangularHiddenLayers(4, 16)
        self.nnm.initializeNeuralNetwork(0.99)
        
        # training
        self.nnm.initializeTraining(0.05, 0.85, 250, 1000, 500)
        self.nnm.train()
        
        # validate by randomly picking inputs
        print("\nValidating:")
        for i in range(10):
            r = round(random.random()*(self.dataSet.getSize()-1))
            self.nnm.checkByIndex(r, True)
        
        # save nn
        #self.exporter.saveNetwork(self.nnm, "/log/model.ckpt")
        self.nnm.close()
        
    def testShuffle(self):
        filename = "controllers/dcdc/controller" # for smaller network use simple
        self.staticController = self.importer.readStaticController(filename)
        
        self.dataSet.readSetFromController(self.staticController)
        #self.dataSet.formatToBinary()
        print(self.dataSet.getBatch(self.dataSet.getSize(),0))
        self.dataSet.shuffle()
        print(self.dataSet.getBatch(self.dataSet.getSize(),0))

cotonn = COTONN()
#cotonn.testShuffle()
cotonn.run()

