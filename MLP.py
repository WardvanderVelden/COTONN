import tensorflow as tf
import NeuralNetworkManager

# The MLP class as presented in this file is a bare bone object oriented approach to creating an MLP
# for the purpose of capturing the behaviour of a static controller (table). 
class MLP:   
        def __init__(self):
            self.debug_mode = False
      
            # Network parameters
            self.layers = []
            self.num_layers = 0
            self.keep_prob = 1
            
            # Neural network general variables (MLP)
            self.s = None 
            self.u = None
            
            self.weights = []
            self.biases = []
            
            self.network = None 
            
            # Neural network training variables
            self.squared_delta = None
            self.loss_function = None
            self.train_function = None
            
            # Tensorflow specific
            self.session = tf.Session()
            
        # Setters
        def setDebugMode(self, value): self.debug_mode = value
        def setNeurons(self, layers):
            self.num_layers = len(layers)
            self.layers = layers
        def setKeepProbability(self, value): self.keep_prob = value
      
        # Getters
        def getNumLayers(self): return self.num_layers
        def getLayers(self): return self.layers
        def getKeepProbability(self): return self.keep_prob
        
        
        # Initialize network function which intializes an initial network with random weights and biases
        def initializeNetwork(self, activation):
            # Initialize tensors
            self.x = tf.placeholder(tf.float32, [None, self.layers[0]])
            self.y = tf.placeholder(tf.float32, [None, self.layers[-1]])
            
            self.weights = [None]*(self.num_layers - 1)
            self.biases = [None]*(self.num_layers - 1)
            
            for i in range(self.num_layers - 1):
                self.weights[i] = tf.Variable(tf.random_normal([self.layers[i], self.layers[i+1]]),tf.float32)
                self.biases[i]= tf.Variable(tf.random_normal([self.layers[i+1]]),tf.float32)

            # Define network
            tf_layers = [None]*(self.num_layers - 1)   
            tf_layers[0] = tf.nn.dropout(tf.sigmoid(tf.add(tf.matmul(self.x, self.weights[0]), self.biases[0])),self.keep_prob) # input layer
            for i in range(1, self.num_layers - 1):
                tf_layers[i] = tf.nn.dropout(tf.sigmoid(tf.add(tf.matmul(tf_layers[i-1], self.weights[i]), self.biases[i])),self.keep_prob)
                
            self.network = tf_layers[-1]
            
            self.session.run(tf.global_variables_initializer())
            
            return self.network
        
        
        # Initialize loss function
        def initializeLossFunction(self):
            self.squared_delta = tf.square(self.network - self.y)
            self.loss_function = tf.reduce_sum(self.squared_delta)
            
            return self.loss_function
        
        
        # Intialize training function
        def initializeTrainFunction(self, function, learning_rate):
            if(function == NeuralNetworkManager.NNOptimizer.Gradient_Descent):
                self.train_function = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss_function)
                print("Training method: Gradient Descent")
            elif(function == NeuralNetworkManager.NNOptimizer.Adadelta):
                self.train_function = tf.train.AdadeltaOptimizer(learning_rate).minimize(self.loss_function)
                print("Training method: AdaDelta")
            elif(function == NeuralNetworkManager.NNOptimizer.Adagrad):
                self.train_function = tf.train.AdagradOptimizer(learning_rate).minimize(self.loss_function)
                print("Training method: Adagrad")
            elif(function == NeuralNetworkManager.NNOptimizer.Adam):
                self.train_function = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_function)
                print("Training method: Adam")
            else:
                self.train_function = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss_function)
                print("Training method: Gradient Descent")
     
            self.session.run(tf.global_variables_initializer())
            
            return self.train_function
        
        
        # Training step with a batch
        def trainStep(self, batch):
            # do one training step using the batch
            self.session.run(self.train_function, {self.x: batch[0], self.y: batch[1]})
            return self.session.run(self.loss_function, {self.x: batch[0], self.y: batch[1]})
            
        
        # Estimator function which estimates the desired outcome based on an input
        def estimate(self, x):
            return self.session.run(self.network, {self.x: x})
            
        
        # Close function which closes tensorflow session
        def close(self):
            self.session.close()
            
            
            
