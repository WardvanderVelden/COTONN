import tensorflow as tf

import time

"""General info on saving: http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/  """

class Exporter:
      
      def __init__(self):
            self.save_location = ('./tmp/saves/model'
      
      #def getsave(self) return self.save_function
      
      def saveNetwork(self, session, path):
            # Create a saver
            self.network_saver = tf.train.Saver()
            self.save_location = path
            # Save the given session      
            self.network_saver.save(session, self.save_location)
            print("Model saved in path: %s" % self.save_location)
            return