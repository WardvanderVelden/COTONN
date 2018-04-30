
# Importer class which will be responsible for reading in controller files and formatting them 
# such that they can be read into PlainControllers or BddControllers and used as neural network
# training data
class Importer:
	def __init__(self):
		self.filename = None


	# Read a plain controller into a format with which we can work in python
	def readPlainController(self, filename):
		self.filename = filename

		file = open(filename + ".scs",'r')

		if(file == None):
			print("Unable to open " + filename + ".scs")

	# Read a bdd controller
	def readBddController(self, filename):
		self.filename = filename

		bdd = open(filename + ".bdd",'r')
		support = open(filename + ".scs", 'r')

		if(bdd == None):
			print("Unable to open " + filename + ".bdd")

		if(support == None):
			print("Unable to open " + filename + ".scs")
