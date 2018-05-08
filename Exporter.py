import tensorflow as tf

"""General info on saving: http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/  """

class Exporter:
      
      def saveVariables(self, session, variables):
            # Create a saver
            saver = tf.train.Saver(variables)
            # Save the given session      
            save_path = saver.save(session, "COTONN_model_variables")
            print("Variables saved in path: %s" % save_path)
      
      def saveNetwork(self, nnm, model_path):
            # Create a saver
            
            # Save the given session      
            save_path = nnm.nn.saver.save(nnm.nn.session, model_path)
            print("Model saved in path: %s" % save_path)




