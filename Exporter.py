#import tensorflow as tf

"""General info on saving: http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/  """

# Exporter class responsible for exporting to files
class Exporter:
      def saveNetwork(self, nnm, filename):
          nnm.save(filename)
          print("Network saved")


