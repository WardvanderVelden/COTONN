
# PlainController class which will hold the controller that can then be accessed in order to read training
# data for the neural network
class PlainController:
	def __init__(self):
		self.state_space_dim = None
		self.state_space_etas = None
		self.state_space_borders = None

		self.input_space_dim = None
		self.input_space_etas = None
		self.input_space_borders = None

		self.state_space = None
		self.input_space = None

	# Return state space id from discrete state variables
	def getIdFromState(self, state):
		return 1

	# Return discrete state variables from id
	def getStateFromId(self, id):
		return 1

	# Return input at state space id
	def getInputFromId(self, id):
		return self.input_space[id]

	# Return input at state space from discrete state variables
	def getInputFromState(self, state):
		return getIdFromState(self.getInputFromId(state)) 


	# Initialize a plain controller
	def initializePlainController(self, ssd, sse, ssb, isd ,ise, isb, ss, iss):
		self.state_space_dim = ssd
		self.state_space_etas = sse
		self.state_space_borders = ssb
		self.state_space = ss

		self.input_space_dim = isd
		self.input_space_etas = ise
		self.input_space_border = isb
		self.input_space = iss




