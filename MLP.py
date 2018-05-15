import tensorflow as tf
import os
import NeuralNetworkManager

# Disable session clear debug log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# The MLP class as presented in this file is a bare bone object oriented approach to creating an MLP
# for the purpose of capturing the behaviour of a static controller (table). 
class MLP:   
        def __init__(self):
            self.debug_mode = False
      
            # Network parameters
            self.layers = []
            self.num_layers = 0
            self.keep_prob_float = 1.0
            self.activation_function = None
            
            # Neural network general variables (MLP)            
            self.weights = []
            self.biases = []
            
            self.predictor = None
            
            # Neural network training variables
            self.loss_function = None
            self.train_function = None
            
            # Tensorflow specific
            tf.reset_default_graph()
            self.session = tf.Session()
            
            self.tensorboard_log_path = './tmp/log/test'
            "--> tensorboard --logdir=studie/wb 3e jaar/bep/cotonn/tmp/log/test "
            
        # Setters
        def setDebugMode(self, value): self.debug_mode = value
        def setNeurons(self, layers):
            self.num_layers = len(layers)
            self.layers = layers
        def setKeepProbability(self, value): self.keep_prob_float = value
        def setActivationFunction(self, value): self.activation_function = value
      
        # Getters
        def getNumLayers(self): return self.num_layers
        def getLayers(self): return self.layers
        def getKeepProbability(self): return self.keep_prob_float
        def getActivationFunction(self): return self.activation_function
        
        
        # Initialize network function which intializes an initial network with random weights and biases
        def initializeNetwork(self, nnm):
            with tf.name_scope('Input'):
                  self.x = tf.placeholder(tf.float32, [None, self.layers[0]], name='InputStates')
                  self.y = tf.placeholder(tf.float32, [None, self.layers[-1]], name='InputLabels')
                  self.keep_prob = tf.placeholder(tf.float32, name='Keep_Prob')

            with tf.name_scope('Input_Layer'):
                  layer = tf.layers.dense(inputs=self.x, units=self.layers[1], activation=self.activationFunction())
                  layer = tf.layers.dropout(inputs=layer, rate=self.keep_prob)

            with tf.name_scope('Hidden_Layers_'):
                  for i in range(1, self.num_layers - 1):
                      layer = tf.layers.dense(inputs=layer, units=self.layers[i+1], activation=self.activationFunction())
                      layer = tf.layers.dropout(inputs=layer, rate=self.keep_prob)

            with tf.name_scope('Predictor'):
                  self.predictor = layer

            self.merged = tf.summary.merge_all() # This argument can not be None, as it will result in an error.
            self.session.run(tf.global_variables_initializer())
            return self.predictor
        
        
        # Activation function
        def activationFunction(self):
            with tf.name_scope('Activation'):
                  if self.activation_function == NeuralNetworkManager.NNActivationFunction.Sigmoid:
                      return tf.sigmoid
                  if self.activation_function == NeuralNetworkManager.NNActivationFunction.Relu:
                      return tf.nn.relu
                  if self.activation_function == NeuralNetworkManager.NNActivationFunction.tanh:
                      return tf.tanh
                  else: 
                      return tf.sigmoid
 
        # Initialize loss function
        def initializeLossFunction(self):
            with tf.name_scope('Loss_Function'):
                  # big difference in learning rate between loss functions
                  #self.loss_function = tf.losses.sigmoid_cross_entropy(self.y, self.predictor)
                  self.loss_function = tf.losses.log_loss(self.y, self.predictor)
                  #self.loss_function = tf.losses.mean_squared_error(self.y, self.predictor)
            return self.loss_function
        
        
        # Intialize training function
        def initializeTrainFunction(self, function, learning_rate):
            with tf.name_scope('Optimizer'):
                  if(function == NeuralNetworkManager.NNOptimizer.Gradient_Descent):
                      self.train_function = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss_function)
                      if(self.debug_mode):
                          print("Training method: Gradient Descent")
                  elif(function == NeuralNetworkManager.NNOptimizer.Adadelta):
                      self.train_function = tf.train.AdadeltaOptimizer(learning_rate).minimize(self.loss_function)
                      if(self.debug_mode):
                          print("Training method: AdaDelta")
                  elif(function == NeuralNetworkManager.NNOptimizer.Adagrad):
                      self.train_function = tf.train.AdagradOptimizer(learning_rate).minimize(self.loss_function)
                      if(self.debug_mode):
                          print("Training method: AdaGrad")
                  elif(function == NeuralNetworkManager.NNOptimizer.Adam):
                      self.train_function = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_function)
                      if(self.debug_mode):
                          print("Training method: Adam")
                  else:
                      self.train_function = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss_function)
                      if(self.debug_mode):
                          print("Training method: Gradient Descent")
           
                  self.session.run(tf.global_variables_initializer())
            return self.train_function
        
        
        # Training step with a batch
        def trainStep(self, batch, merged_summary):
            with tf.name_scope('TRAIN'):
                  acc, summary, loss = self.session.run([self.train_function, merged_summary, self.loss_function],  
                                                        {self.x: batch[0], self.y: batch[1], self.keep_prob: self.keep_prob_float})  
            return loss, summary
            
        
        # Estimator function which estimates the desired outcome based on an input
        def estimate(self, x):
            return self.session.run(self.predictor, {self.x: x, self.keep_prob: 1.0})
            
  
        # Close function which closes tensorflow session
        def close(self):
            self.session.close()
            
            
            
