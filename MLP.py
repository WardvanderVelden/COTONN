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
            self.session = tf.Session()
            self.saver = None
            
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
        
        def getSession(self): return self.session
        
        
        # Initialize network function which intializes an initial network with random weights and biases
        def initializeNetwork(self):
            self.x = tf.placeholder(tf.float32, [None, self.layers[0]])
            self.y = tf.placeholder(tf.float32, [None, self.layers[-1]])
            self.keep_prob = tf.placeholder(tf.float32)

            layer = tf.layers.dense(inputs=self.x, units=self.layers[1], activation=self.activationFunction())
            layer = tf.layers.dropout(inputs=layer, rate=self.keep_prob)

            for i in range(1, self.num_layers - 1):
                layer = tf.layers.dense(inputs=layer, units=self.layers[i+1], activation=self.activationFunction())
                layer = tf.layers.dropout(inputs=layer, rate=self.keep_prob)

            self.predictor = layer

            self.session.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
        
        
        # Activation function
        def activationFunction(self):
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
            self.loss_function = tf.losses.log_loss(self.y, self.predictor)
            
            return self.loss_function
        
        
        # Intialize training function
        def initializeTrainFunction(self, function, learning_rate):
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
        def trainStep(self, batch):
            # do one training step using the batch
            self.session.run(self.train_function, {self.x: batch[0], self.y: batch[1], self.keep_prob: self.keep_prob_float})
            return self.session.run(self.loss_function, {self.x: batch[0], self.y: batch[1], self.keep_prob: self.keep_prob_float})
            
        
        # Estimator function which estimates the desired outcome based on an input
        def estimate(self, x):
            return self.session.run(self.predictor, {self.x: x, self.keep_prob: 1.0})
            
        
        # Save
        def save(self, filename):
            self.saver.save(self.session, filename)
        
        # Close function which closes tensorflow session
        def close(self):
            self.session.close()
            
            
            
