import tensorflow as tf

"""General info on saving: http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/  """

class Exporter:
      
      def saveVariables(self, session, variables):
            # Create a saver
            saver = tf.train.Saver(variables)
            # Save the given session      
            save_path = saver.save(session, "COTONN_model_variables")
            print("Variables saved in path: %s" % save_path)
      
      def saveNetwork(self, session):
            # Create a saver
            saver = tf.train.Saver()
            # Save the given session      
            save_path = saver.save(session, "COTONN_model")
            print("Model saved in path: %s" % save_path)




