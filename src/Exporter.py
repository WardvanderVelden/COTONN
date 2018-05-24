import tensorflow as tf
import numpy as np

# Exporter class responsible for exporting to files
class Exporter:
    def __init__(self, version):
        self.save_location = "./saves/"
        self.version = version
      
      
    def setSaveLocation(self, value): self.save_location = value
      
      
    # Save network graph
    def saveNetwork(self, nnm):
          # Create a saver
        self.network_saver = tf.train.Saver()
        session = nnm.nn.session
            
        # Save the given session      
        self.network_saver.save(session, self.save_location + "model")  
        print("\nModel saved to path: " + self.save_location + "model")
    
    
    # Save network variables
    def saveVariables(self, nnm, list_variables): 
        # Create a saver
        self.saver = tf.train.Saver(list_variables)
        session = nnm.nn.session
        
        # Save the given session      
        self.saver.save(session, self.save_location + "variable")
        print("Variables saved to path: " + self.save_location + "variable")
    
    
    # Save variables as text
    def saveRawMLP(self, nnm):
        file = open(self.save_location + "nn.txt", "w")
        file.write("COTONN v" + self.version + " raw NN:\n")
        
        session = nnm.nn.session
        layers = nnm.nn.layers
        
        for i in range (len(layers) - 1):
            with tf.variable_scope("layer_" + str(i), reuse=True):
                weight = tf.get_variable("kernel")
                bias = tf.get_variable("bias")
                
                weight_eval = session.run(weight)
                bias_eval = session.run(bias)
                
                file.write("\nW" + str(i) + "\n")
                
                np.savetxt(file, weight_eval)
                file.write("\nb" + str(i) + "\n")
                np.savetxt(file, bias_eval)
                
        file.close()
        print("Raw MLP saved to path: " + self.save_location + "nn.txt")
        
    
    # Save neural network in a very simple format that is executable as a MATLAB script
    def saveMatlabMLP(self, controller, nnm):
        file = open(self.save_location + "nn.m", "w")
        
        session = nnm.nn.session
        layers = nnm.nn.layers
        
        file.write("s_eta = [")
        for x in controller.getStateSpaceEtas(): file.write(x + " ")
        file.write("];\n")
        
        file.write("s_ll = [")
        for x in controller.getStateSpaceLowerLeft(): file.write(x + " ")
        file.write("];\n")
        
        file.write("s_ur = [")
        for x in controller.getStateSpaceUpperRight(): file.write(x + " ")
        file.write("];\n")
        
        file.write("\nu_eta = [")
        for x in controller.getInputSpaceEtas(): file.write(x + " ")
        file.write("];\n")
        
        file.write("u_ll = [")
        for x in controller.getInputSpaceLowerLeft(): file.write(x + " ")
        file.write("];\n")
        
        file.write("u_ur = [")
        for x in controller.getInputSpaceUpperRight(): file.write(x + " ")
        file.write("];\n")

        
        for i in range (len(layers) - 1):
            with tf.variable_scope("layer_" + str(i), reuse=True):
                weight = tf.get_variable("kernel")
                bias = tf.get_variable("bias")
                
                weight_eval = session.run(weight)
                bias_eval = session.run(bias)
                
                file.write("\nW{"+ str(i+1) + "} = [")
                np.savetxt(file, weight_eval)
                file.write("];\n")
                
                file.write("\nb{" + str(i+1) + "} = [")
                np.savetxt(file, bias_eval)
                file.write("];\n")
                
        file.close()
        print("Matlab MLP saved to path: " + self.save_location + "nn.m")
        
        
    # Save as a binary dump (smallest representation)
    def saveBinary(self, nnm):
        file = open(self.save_location + "nn.cot","wb")
        
        session = nnm.nn.session
        layers = nnm.nn.layers
        
        for i in range (len(layers) - 1):
            with tf.variable_scope("layer_" + str(i), reuse=True):
                weight = tf.get_variable("kernel")
                bias = tf.get_variable("bias")
                
                weight_eval = session.run(weight)
                bias_eval = session.run(bias)
                
                weight_shape = session.run(tf.shape(weight))
                bias_shape = session.run(tf.shape(bias))
                
                file.write(int(weight_shape[0]).to_bytes(4, "little"))
                file.write(int(weight_shape[1]).to_bytes(4, "little"))
                file.write(weight_eval.tostring())
                
                file.write(int(bias_shape[0]).to_bytes(4, "little"))
                file.write(bias_eval.tostring())
                
        print("Binary saved to path: " + self.save_location + "nn.cot")
        file.close()                
    
    
    # Save wrong states as a text file
    def saveWrongStates(self, wrong_states):
        file = open(self.save_location + "wrong_states.txt", "w")
        file.write("COTONN v" + self.version + " Wrong states (#" + str(len(wrong_states)) + "): \n")
        
        if(len(wrong_states)== 0): return        
                
        for i in range(len(wrong_states)):
            file.write(str(wrong_states[i]) + "\n")
            
        file.close()
        
        print("Wrong states saved to path: " + self.save_location + "wrong_states.txt")
