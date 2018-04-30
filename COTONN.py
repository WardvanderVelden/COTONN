from Importer import Importer
from PlainController import PlainController
from BddController import BddController
from NeuralNetworkManager import NeuralNetworkManager

# Main class from which all functions are called
class COTONN:
	def __init__(self):
		self.importer = Importer()
		self.plainController = PlainController()
		self.nn = NeuralNetworkManager()

	def test(self):
		print("Test succesful!")
		print(self.plainController.getIdFromState(None))

cotonn = COTONN()
cotonn.test()