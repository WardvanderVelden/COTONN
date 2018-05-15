import tensorflow as tf

"""General info on saving: http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/  """

# Exporter class responsible for exporting to files
class Exporter:
    def __init__(self):
        self.save_location = "./nn/saves/model"
      
      
    def saveNetwork(self, session, path):
        # Create a saver
        self.network_saver = tf.train.Saver()
        self.save_location = path
            
        # Save the given session      
        self.network_saver.save(session, self.save_location)
        print("\nModel saved in path: %s" % self.save_location)
        return