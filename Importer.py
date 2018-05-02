from PlainController import PlainController
import numpy as np

# Importer class which will be responsible for reading in controller files and formatting them 
# such that they can be read into PlainControllers or BddControllers and used as neural network
# training data
class Importer:
      def __init__(self):
        self.filename = None
        self.debug_mode = False
        
    def setDebugMode(self, value):
        self.debug_mode = value

      # Read a plain controller into a format with which we can work in python
      def readPlainController(self, filename):
            self.filename = filename
            file = open(filename + ".scs",'r')

            if(file == None):
                  print("Unable to open " + filename + ".scs")
            
            self.raw_str = file.read()   
            con = PlainController()
            
            # Retrieve parameters of the STATE_SPACE
            # Retrieve the number of dimensions from the text file
            con.state_space_dim = self.getLine("MEMBER:DIM\n", "\n#VECTOR:ETA\n")
            print("This is the dimension of the SPACE_STATE:\n", con.state_space_dim)
            
            # Retrieve the number of eta's from the text file
            con.state_space_etas = self.getLine(("#VECTOR:ETA\n#BEGIN:" + con.state_space_dim + "\n"), "\n#END")
            con.state_space_etas = con.state_space_etas.split('\n')  
            print("These are the etas of the SPACE_STATE:\n", con.state_space_etas)
            
            # Retrieve the borders from the text file
            state_border1 = self.getLine("#VECTOR:LOWER_LEFT\n#BEGIN:" + con.state_space_dim + "\n", "#END\n#VECTOR:UPPER_RIGHT")
            state_border2 = self.getLine("#VECTOR:UPPER_RIGHT\n#BEGIN:" + con.state_space_dim + "\n", "\n#END\n#SCOTS:INPUT_SPACE")
            state_space_borders = state_border1 + state_border2
            state_space_borders = state_space_borders.split('\n')
            state_space_borders = np.array(state_space_borders)
            shape_state_space = (int(len(state_space_borders)/2), 2)
            con.state_space_borders = state_space_borders.reshape(shape_state_space)
            print("These are the borders of the SPACE_STATE:\n", con.state_space_borders)
            
            
            # Now the same as above for the INPUT_SPACE
            self.raw_str = self.raw_str[self.raw_str.index("#SCOTS:INPUT_SPACE"):]
            
            con.input_space_dim = self.getLine("#TYPE:UNIFORMGRID\n#MEMBER:DIM\n", "\n#VECTOR:ETA")
            print("This is the dimension of the INPUT_SPACE:\n", con.input_space_dim)
            
            # Retrieve the number of eta's from the text file
            con.input_space_etas = self.getLine(("#VECTOR:ETA\n#BEGIN:" + con.input_space_dim + "\n"), "\n#END")
            con.input_space_etas = con.input_space_etas.split('\n')  
            print("These are the etas of the INPUT_SPACE:\n", con.input_space_etas)
            
            # Retrieve the borders from the text file
            input_border1 = self.getLine("#VECTOR:LOWER_LEFT\n#BEGIN:" + con.input_space_dim + "\n", "#END\n#VECTOR:UPPER_RIGHT")
            input_border2 = self.getLine("#VECTOR:UPPER_RIGHT\n#BEGIN:" + con.input_space_dim + "\n", "\n#END\n#TYPE:WINNINGDOMAIN")
            input_space_borders = input_border1 + input_border2
            input_space_borders = input_space_borders.split('\n')   
            input_space_borders = np.array(input_space_borders)
            shape_input_space = (int(len(input_space_borders)/2), 2)
            con.input_space_borders = input_space_borders.reshape(shape_input_space)             
            print("These are the borders of the INPUT_SPACE:\n", con.input_space_borders)
            
            # Now we extract the matrix data from the text file
            matrix_data = self.raw_str[self.raw_str.index("#MATRIX:DATA"):]
            matrix_data = matrix_data.split(' \n')                                               
            matrix_data = matrix_data[2:]
            print(matrix_data[:20])
            con.state_space = []
            con.input_space = []
            for i in range(len(matrix_data)-1):
                  single_string = matrix_data[i]
                  split_data = single_string.split()
                  con.state_space.append(split_data[0])
                  con.input_space.append(split_data[1])
            #print(con.state_space[:100], con.input_space[:100])
            return con
      
      # Function used to retrieve specific information from the text file. 
      def getLine(self, left, right):
            dim_left = self.raw_str.index(left)+len(left)
            dim_right = self.raw_str.index(right)
            line = self.raw_str[dim_left:dim_right]
            #print("Line found = ", line)
            return line
