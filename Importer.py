from StaticController import StaticController
import numpy as np
import tensorflow as tf


# Importer class which will be responsible for reading in controller files and formatting them 
# such that they can be read into PlainControllers or BddControllers and used as neural network
# training data
class Importer:
      def __init__(self):
            self.filename = None
            self.debug_mode = False
            self.restore_mode = False
        
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
            state_space_lower_left = self.getLine("#VECTOR:LOWER_LEFT\n#BEGIN:" + ctl.getStateSpaceDim() + "\n", "\n#END\n#VECTOR:UPPER_RIGHT").split('\n')
            state_space_upper_right = self.getLine("#VECTOR:UPPER_RIGHT\n#BEGIN:" + ctl.getStateSpaceDim() + "\n", "\n#END\n#SCOTS:INPUT_SPACE").split('\n')
            ctl.setStateSpaceLowerLeft(state_space_lower_left)
            ctl.setStateSpaceUpperRight(state_space_upper_right)
            if(self.debug_mode):                
                print("This is the lower left of the SPACE_STATE:\n", ctl.getStateSpaceLowerLeft())
                print("This is the upper right of the SPACE_STATE:\n", ctl.getStateSpaceUpperRight())
            
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
            input_space_lower_left = self.getLine("#VECTOR:LOWER_LEFT\n#BEGIN:" + ctl.getInputSpaceDim() + "\n", "\n#END\n#VECTOR:UPPER_RIGHT").split('\n')
            input_space_upper_right = self.getLine("#VECTOR:UPPER_RIGHT\n#BEGIN:" + ctl.getInputSpaceDim() + "\n", "\n#END\n#TYPE:WINNINGDOMAIN").split('\n')
            ctl.setInputSpaceLowerLeft(input_space_lower_left)
            ctl.setInputSpaceUpperRight(input_space_upper_right)
            if(self.debug_mode):                
                print("This is the lower left of the INPUT_SPACE:\n", ctl.getInputSpaceLowerLeft())
                print("This is the upper right of the INPUT_SPACE:\n", ctl.getInputSpaceUpperRight())
           
            # Now we extract the matrix data from the text file
            matrix_data_left = self.raw_str.index("#MATRIX:DATA")
            matrix_str = self.raw_str[matrix_data_left:]
            matrix_str = matrix_str[:matrix_str.index("\n#END")]                                     
                                                  
            matrix_data = matrix_str.split('\n')                                               
            matrix_data = matrix_data[2:]
            for i in range(len(matrix_data)):
                split_data = matrix_data[i].split(' ')
                s = int(split_data[0].replace(' ',''))
                u = int(split_data[1].replace(' ',''))
                ctl.setStateInput(s, u)
                      
            # Calculate controller size
            ctl.setSize()
                  
            return ctl
      
      # Function used to retrieve specific information from the text file. 
      def getLine(self, left, right):
            dim_left = self.raw_str.index(left)+len(left)
            dim_right = self.raw_str.index(right)
            line = self.raw_str[dim_left:dim_right]
            return line
      
      def restoreNetwork(self, nnm,  meta_path):
            self.restore_mode = True
            self.network_saver = tf.train.Saver()
            session = nnm.nn.session

            self.network_saver.restore(session, meta_path)
            print("Network restored from path: " + meta_path)
            return self.network_saver, self.restore_mode

      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
