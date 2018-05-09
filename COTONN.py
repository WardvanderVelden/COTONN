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
        
        self.fullSet = DataSet()
        self.subSet = DataSet()
        
        self.debug_mode = True
        
        self.importer.setDebugMode(False)
        self.nnm.setDebugMode(self.debug_mode)


    # Test function to automatically convert a plain controller to a simple MLP network
    def run(self):      
        print("COTONN v0.3\n")
        
        # read static controller
        filename = "controllers/vehicle/controller" # for smaller network use simple
        self.staticController = self.importer.readStaticController(filename)
        
        # define dataset
        self.fullSet.readSetFromController(self.staticController)
        self.fullSet.formatToBinary()
        self.subSet.readSubsetFromController(self.staticController, 0.02)
        self.subSet.formatToBinary()
        
        # specify neural network
        self.nnm.setType(NNTypes.MLP)
        self.nnm.setTrainingMethod(NNOptimizer.Adam)
        self.nnm.setActivationFunction(NNActivationFunction.Sigmoid)
        self.nnm.setDataSet(self.subSet)
        
        self.nnm.setKeepProbability(1.0)
        self.nnm.rectangularHiddenLayers(2, 8)
        self.nnm.initializeNeuralNetwork()
        
        # training
        self.nnm.initializeTraining(0.015, 0.99, 100, 1000)
        self.nnm.train()
        
        # check fitness with the full set
        self.nnm.setDataSet(self.subSet)
        fit = self.nnm.checkFitness()
        print("\nFullset fitness: " + str(float("{0:.3f}".format(fit))))
        
        print("\nValidating:")
        for i in range(10):
            r = round(random.random()*(self.subSet.getSize()-1))
            self.nnm.checkByIndex(r, True)
        
        # save nn
        self.exporter.saveNetwork(self.nnm, "./nn/model.ckpt")
        
        # close session
        self.nnm.close()

cotonn = COTONN()
cotonn.run()

