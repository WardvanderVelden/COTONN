import tensorflow as tf
#from NeuralNetworkManager import NNActivationFunction, NNOptimizer
import NeuralNetworkManager
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

class MLP:   
        #tf.logging.set_verbosity(tf.logging.INFO)
        def __init__(self):
            self.learning_rate = 0.1
            self.num_steps = 50
            self.batch_size = 256
            self.display_step = 200
            
            self.num_inputs = 0
            self.num_outputs = 0
            self.h_layers = []
            
            self.activation_type = None
            self.loss_function = None
            self.optimizer_type = None
            
            self.neural_network = None # contains the layers of the neural network
     
        # Setters
        def setLearningRate(self, value): self.learning_rate = value
        def setNumSteps(self, value): self.num_steps = value
        def setBatchSize(self, value): self.batch_size = value
        def setDisplayStep(self, value): self.display_step = value
        def setHiddenNeuronsLayers(self, value): self.h_layers = value
        def setNumInputs(self, value): self.num_inputs = value
        def setNumOutputs(self, value): self.num_outputs = value
        def setActivationType(self, value): self.activation_type = value
        def setLossFunction(self, value): self.loss_function = value
        def setOptimizer(self, value): self.optimizer_type = value
      
        def setNeurons(self, layers):
            self.num_inputs = int(layers[0])
            self.num_outputs = int(layers[-1])
          
            self.h_layers = [0]*(len(layers) - 2)
          
            for i in range(0, len(layers) - 2):
                self.h_layers[i] = layers[i+1]
      
        # Getters
        def getLearningRate(self): return self.learning_rate
        def getNumSteps(self): return self.num_steps
        def getBatchSize(self): return self.batch_size
        def getDisplayStep(self): return self.display_step
        def getHiddenNeuronsLayers(self): return self.h_layers 
        def getNumInputs(self): return self.num_inputs
        def getNumOutputs(self): return self.num_outputs
        def getActivationType(self): return self.activation_type 
        def getOptimizer(self): return self.optimizer_type
        
        def getLossFunction(self, targets, predictions):
            if self.loss_function == None:
                self.loss_function = tf.losses.mean_squared_error(targets, predictions)
            return self.loss_function


        def getTrainingInputs(self, training_data):
            training_states = tf.constant(training_data.states)
            training_labels = tf.constant(training_data.labels)
            return training_states, training_labels
      
        def getTestInputs(self, test_data):
            test_states = tf.constant(test_data.states)
            test_labels = tf.constant(test_data.labels)
            return test_states, test_labels

        # Initialize neural network
        def initialize(self): 
            
#            in_layer = tf.layers.dense(self.num_inputs, self.h_layers[0])
#            in_layer = self.activationFunction(in_layer)
#            
#            hidden_layers = []
#            hidden_layers.append(tf.layers.dense(in_layer, self.h_layers[0]))
#            for i in range(1, len(self.n_layer)):
#                  hidden_layers.append(tf.layers.dense(hidden_layers[i-1], self.h_layers[i]))  
#            hidden_layers = self.activationFunction(hidden_layers)
#            
#            out_layer = tf.layers.dense(hidden_layers[-1], self.num_outputs)
#            out_layer = self.activationFunction(self.out_layer)
#            
#            self.neural_network = out_layer
                      
        # Generate model function     
        def generateModelFunction(self, features, targets, mode, params):
            # Logic to do the following:
            # 1. Configure the model via TensorFlow operations
            # 2. Define the loss function for training/evaluation
            # 3. Define the training operation/optimizer
            # 4. Generate predictions
            # 5. Return predictions/loss/train_op/eval_metric_ops in ModelFnOps object
            
            # Reshape output layer to 1-dim Tensor to return predictions
            predictions = tf.reshape(self.neural_network, [-1])
            
            # Calculate loss using mean squared error
            loss = self.getLossFunction(targets, predictions)
            
            # Calculate root mean squared error as additional eval metric
            eval_metric_ops = {"rmse": tf.metrics.root_mean_squared_error(
                              tf.cast(targets, tf.float64), predictions)}
            
            train_op = tf.contrib.layers.optimize_loss(
                        loss = loss,
                        global_step = tf.contrib.framework.get_global_step(),
                        learning_rate = self.learning_rate,
                        optimizer = self.optimizerFunction)
            return model_fn_lib.ModelFnOps(mode, predictions, loss, train_op, eval_metric_ops)
      
        def activationFunction(self, layer):
            if self.activation_type == None:
                self.activationFunction = tf.nn.linear(layer)
            if self.activation_type == NeuralNetworkManager.NNActivationFunction.Relu:
                self.activationFunction = tf.nn.relu(layer)
            if self.activation_type == NeuralNetworkManager.NNActivationFunction.Sigmoid:
                self.activationFunction = tf.nn.relu(layer)
            return self.activationFunction
      
        def optimizerFunction(self):
            optimizer_function = ""
            if self.optimizer_type == None:
                optimizer_function = "SGD"
            if self.optimizer_type == NeuralNetworkManager.NNOptimizer.Gradient_Descent:
                optimizer_function = "SGD"
            if self.optimizer_type == NeuralNetworkManager.NNOptimizer.Adagrad:
                optimizer_function = "Adagrad"
            if self.optimizer_type == NeuralNetworkManager.NNOptimizer.Adadelta:
                optimizer_function = "Adadelta"
            if self.optimizer_type == NeuralNetworkManager.NNOptimizer.Adam:
                optimizer_function = "Adam"
            if self.optimizer_type == NeuralNetworkManager.NNOptimizer.Ftrl:
                optimizer_function = "Ftrl"
            if self.optimizer_type == NeuralNetworkManager.NNOptimizer.Momentum:
                optimizer_function = "Momentum"
            if self.optimizer_type == NeuralNetworkManager.NNOptimizer.RMSProp:
                optimizer_function = "RMSProp"
            return optimizer_function  

        # Train neural network
        def train(self, data):
            nn = tf.contrib.learn.Estimator(model_fn=self.generateModelFunction, params = self.learning_rate)
            nn.fit(input_fn=self.get_train_inputs, steps=self.num_steps)
            
            ev = nn.evaluate(input_fn=self.getTestInputs, steps=1)
            print("Loss: %s" % ev["loss"])
            print("Root Mean Squared Error: %s" % ev["rmse"])
 
        # This still needs to be properly worked out
        def test(self, neural_network):
            # Prediction set is the data we want to predict, so some examples
            test_nn = neural_network
            test_estimation = test_nn.predict(x=self.test_states, as_iterable=True)
            for i, p in enumerate(test_estimation):
                  print("Prediction %s: %s" % (i + 1, p[""]))
      
















