import tensorflow as tf
import numpy as np

# Exporter class responsible for exporting to files
class Exporter:
    def __init__(self, version):
        self.save_location = "./nn/"
        self.version = version
      
      
    def setSaveLocation(self, value): self.save_location = value
      
      
    def saveNetwork(self, nnm):
          # Create a saver
        self.network_saver = tf.train.Saver()
        session = nnm.nn.session
            
        # Save the given session      
        self.network_saver.save(session, self.save_location + "model")  
        print("\nModel saved in path: " + self.save_location + "model")
    
    
    def saveVariables(self, nnm, list_variables): 
        # Create a saver
        self.saver = tf.train.Saver(list_variables)
        session = nnm.nn.session
        
        # Save the given session      
        self.saver.save(session, self.save_location + "variable")
        print("\nVariables saved in path: " + self.save_location + "variable")
    
    
    def saveRawMLP(self, nnm):
        file = open(self.save_location + "nn.cot", "w")
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
        print("\nRaw MLP saved to path: " + self.save_location + "nn.cot")
        
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
        print("\nMatlab MLP saved to path: " + self.save_location + "nn.m")
    
    
    def saveWrongStates(self, wrong_states):
        file = open(self.save_location + "wrong_states.txt", "w")
        file.write("COTONN v" + self.version + " Wrong states (#" + str(len(wrong_states)) + "): \n")
        
        if(len(wrong_states)== 0): return        
                
        for i in range(len(wrong_states)):
            file.write(str(wrong_states[i]) + "\n")
            
        file.close()
        
        print("\nWrong states saved to path: " + self.save_location + "wrong_states.txt")

