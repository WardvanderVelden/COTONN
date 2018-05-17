import tensorflow as tf

# Exporter class responsible for exporting to files
class Exporter:
    def __init__(self):
        self.save_location = "./nn/saves/model"
      
      
    def saveNetwork(self, NeuralNetworkManager, meta_path):
          # Create a saver
        self.network_saver = tf.train.Saver()
        self.save_location = meta_path
        self.nnm = NeuralNetworkManager
        self.session = self.nnm.nn.session
            
        # Save the given session      
        self.network_saver.save(self.session, self.save_location)  
        print("\nModel saved in path: %s" % self.save_location)
    
    def saveVariables(self, NeuralNetworkManager, meta_path, list_variables): 
        # Create a saver
        self.saver = tf.train.Saver(list_variables)
        self.save_location = meta_path
        self.nnm = NeuralNetworkManager
        self.session = self.nnm.nn.session
        
        # Save the given session      
        self.saver.save(self.session, self.save_location)
        print("\nVariables saved in path: %s" % self.save_location)
    
    
    
    