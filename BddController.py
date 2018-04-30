
# BddController class which will hold the controller that can then be accessed in order to read training
# data for the neural network
class BddController:
	def __init__(self):
		self.bdd = None

	# Return state space id from discrete state variables
	def getIdFromState(self, state):
		return 1

	# Return discrete state variables from id
	def getStateFromId(self, id):
		return 1

	# Return input at state space id
	def getInputFromId(self, id):
		return 1

	# Return input at state space from discrete state variables
	def getInputFromState(self, state):
		return getIdFromState(self.getInputFromId(state)) 

