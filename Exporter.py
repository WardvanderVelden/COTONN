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
        self.network_saver.save(session, self.save_location + "model.ckpt")  
        print("\nModel saved in path: " + self.save_location + "model.ckpt")
    
    
    def saveVariables(self, nnm, list_variables): 
        # Create a saver
        self.saver = tf.train.Saver(list_variables)
        session = nnm.nn.session
        
        # Save the given session      
        self.saver.save(session, self.save_location)
        print("\nVariables saved in path: " + self.save_location)
    
    
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
    
    
    def saveWrongStates(self, wrong_states):
        file = open(self.save_location + "wrong_states.txt", "w")
        file.write("COTONN v" + self.version + " Wrong states (#" + str(len(wrong_states)) + "): \n")
        
        for i in range(len(wrong_states)):
            file.write(str(wrong_states[i]) + "\n")
            
        file.close()
        
        print("\nWrong states saved to path: " + self.save_location + "wrong_states.txt")

