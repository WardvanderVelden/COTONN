from StaticController import StaticController
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

      # Read a static controller into a format with which we can work in python
      def readStaticController(self, filename):
            self.filename = filename
            file = open(filename + ".scs",'r')

            if(file == None):
                  print("Unable to open " + filename + ".scs")
            
            self.raw_str = file.read()   
            ctl = StaticController()
            
            # Retrieve parameters of the STATE_SPACE
            # Retrieve the number of dimensions from the text file
            ctl.setStateSpaceDim(self.getLine("MEMBER:DIM\n", "\n#VECTOR:ETA\n"))
            if(self.debug_mode):
                print("This is the dimension of the SPACE_STATE:\n", ctl.getStateSpaceDim())
            
            # Retrieve the number of eta's from the text file 
            ctl.setStateSpaceEtas(self.getLine(("#VECTOR:ETA\n#BEGIN:" + ctl.getStateSpaceDim() + "\n"), "\n#END").split('\n'))
            if(self.debug_mode):
                print("These are the etas of the SPACE_STATE:\n", ctl.getStateSpaceEtas())
            
            # Retrieve the borders from the text file
            state_bound1 = self.getLine("#VECTOR:LOWER_LEFT\n#BEGIN:" + ctl.getStateSpaceDim() + "\n", "#END\n#VECTOR:UPPER_RIGHT")
            state_bound2 = self.getLine("#VECTOR:UPPER_RIGHT\n#BEGIN:" + ctl.getStateSpaceDim() + "\n", "\n#END\n#SCOTS:INPUT_SPACE")
            state_space_bounds = state_bound1 + state_bound2
            state_space_bounds = state_space_bounds.split('\n')
            state_space_bounds = np.array(state_space_bounds)
            shape_state_space = (int(len(state_space_bounds)/2), 2)
            ctl.setStateSpaceBounds(state_space_bounds.reshape(shape_state_space))
            if(self.debug_mode):                
                print("These are the bounds of the SPACE_STATE:\n", ctl.getStateSpaceBounds())
            
            # Now the same as above for the INPUT_SPACE
            self.raw_str = self.raw_str[self.raw_str.index("#SCOTS:INPUT_SPACE"):]
            
            ctl.setInputSpaceDim(self.getLine("#TYPE:UNIFORMGRID\n#MEMBER:DIM\n", "\n#VECTOR:ETA"))
            if(self.debug_mode):
                print("This is the dimension of the INPUT_SPACE:\n", ctl.getInputSpaceDim())
            
            # Retrieve the number of eta's from the text file
            ctl.setInputSpaceEtas(self.getLine(("#VECTOR:ETA\n#BEGIN:" + ctl.getInputSpaceDim() + "\n"), "\n#END").split('\n'))
            if(self.debug_mode):
                print("These are the etas of the INPUT_SPACE:\n", ctl.getInputSpaceEtas())
            
            # Retrieve the borders from the text file
            input_bound1 = self.getLine("#VECTOR:LOWER_LEFT\n#BEGIN:" + ctl.getInputSpaceDim() + "\n", "#END\n#VECTOR:UPPER_RIGHT")
            input_bound2 = self.getLine("#VECTOR:UPPER_RIGHT\n#BEGIN:" + ctl.getInputSpaceDim() + "\n", "\n#END\n#TYPE:WINNINGDOMAIN")
            input_space_bounds = input_bound1 + input_bound2
            input_space_bounds = input_space_bounds.split('\n')   
            input_space_bounds = np.array(input_space_bounds)
            shape_input_space = (int(len(input_space_bounds)/2), 2) 
            ctl.setInputSpaceBounds(input_space_bounds.reshape(shape_input_space))          
            if(self.debug_mode):
                print("These are the bounds of the INPUT_SPACE:\n", ctl.getInputSpaceBounds())
            
            # Now we extract the matrix data from the text file
            matrix_data = self.raw_str[self.raw_str.index("#MATRIX:DATA"):]
            matrix_data = matrix_data.split(' \n')                                               
            matrix_data = matrix_data[2:]
            for i in range(len(matrix_data)-1):
                  single_string = matrix_data[i]
                  split_data = single_string.split()
                  ctl.setStateInput(split_data[0], split_data[1])
                  if(i%1000 == 0 and self.debug_mode):
                      print("SS: " + split_data[0] + " IS: " + split_data[1])
                      
            # Calculate controller size
            ctl.setSize()
                  
            return ctl
      
      # Function used to retrieve specific information from the text file. 
      def getLine(self, left, right):
            dim_left = self.raw_str.index(left)+len(left)
            dim_right = self.raw_str.index(right)
            line = self.raw_str[dim_left:dim_right]
            return line
