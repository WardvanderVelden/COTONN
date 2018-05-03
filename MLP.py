import tensorflow as tf
from NeuralNetworkManager import NNActivationFunction, NNOptimizer
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

class MLP:   
      tf.logging.set_verbosity(tf.logging.INFO)
      def __init__(self):
            self.learning_rate = 0.1
            self.num_steps = 50
            self.batch_size = 256
            self.display_step = 200
            self.n_layer = [256, 256]
            self.num_input = 784
            self.num_classes = 10
            self.activation_type = None
            self.loss_function = None
            self.optimizer_type = None
     
      def setLearningRate(self, value): self.learning_rate = value
      def setNumSteps(self, value): self.num_steps = value
      def setBatchSize(self, value): self.batch_size = value
      def setDisplayStep(self, value): self.display_step = value
      def setNeuronsLayer(self, value): self.n_layer = value
      def setNumInput(self, value): self.num_input = value
      def setNumClasses(self, value): self.num_classes = value
      def setActivationType(self, value): self.activation_type = value
      def setLossFunction(self, value): self.loss_function = value
      def setOptimizer(self, value): self.optimizer_type = value
      
      def getLearningRate(self): return self.learning_rate
      def getNumSteps(self): return self.num_steps
      def getBatchSize(self): return self.batch_size
      def getDisplayStep(self): return self.display_step
      def getNeuronsLayer(self): return self.n_layer 
      def getNumInput(self): return self.num_input
      def getNumClasses(self): return self.num_classes
      def getActivationType(self): return self.activation_type 
      def getLossFunction(self): return self.loss_function
      def getOptimizer(self): return self.optimizer_type


      def getTrainingInputs(self, training_data):
            training_states = tf.constant(training_data.states)
            training_labels = tf.constant(training_data.labels)
            return training_states, training_labels
      
      def getTestInputs(self, test_data):
            test_states = tf.constant(test_data.states)
            test_labels = tf.constant(test_data.labels)
            return test_states, test_labels

      def initializeNeuralNet(self):  #input_features_dict  
            in_layer = tf.layers.dense(self.num_input, self.n_layer[0])
            in_layer = self.activationFunction(in_layer)
            
            hidden_layers = []
            hidden_layers.append(tf.layers.dense(in_layer, self.n_layer[0]))
            for i in range(0, len(self.n_layer)-1):
                  hidden_layers.append(tf.layers.dense(hidden_layers[i-1], self.n_layer[i]))  
            hidden_layers = self.activationFunction(hidden_layers)
            
            out_layer = tf.layers.dense(hidden_layers[-1], self.num_classes)
            out_layer = self.activationFunction(out_layer)
            return out_layer
                      
      def generateModelFunction(self, features, targets, mode, params):
            # Logic to do the following:
            # 1. Configure the model via TensorFlow operations
            # 2. Define the loss function for training/evaluation
            # 3. Define the training operation/optimizer
            # 4. Generate predictions
            # 5. Return predictions/loss/train_op/eval_metric_ops in ModelFnOps object
            
            NeuralNet = self.NeuralNet()
            
            # Reshape output layer to 1-dim Tensor to return predictions
            predictions = tf.reshape(NeuralNet, [-1])
            
            # Calculate loss using mean squared error
            loss = self.lossFunction(targets, predictions)
            
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
            if self.activation_type == NNActivationFunction.Relu:
                  self.activationFunction = tf.nn.relu(layer)
            if self.activation_type == NNActivationFunction.Sigmoid:
                  self.activationFunction = tf.nn.relu(layer)
            return self.activationFunction
      
      def lossFunction(self, targets, predictions):
            if self.Loss_function == None:
                  self.setLossFunction = tf.losses.mean_squared_error(targets, predictions)
            return self.lossFunction

      def optimizerFunction(self, layer):
            if self.optimizer_type == None:
                  self.optimizerFunction = "SGD"
            if self.optimizer_type == NNOptimizer.Gradient_Descent:
                  self.optimizerFunction = "SGD"
            if self.optimizer_type == NNOptimizer.Adagrad:
                  self.optimizerFunction = "Adagrad"
            if self.optimizer_type == NNOptimizer.Adadelta:
                  self.optimizerFunction = "Adadelta"
            if self.optimizer_type == NNOptimizer.Adam:
                  self.optimizerFunction = "Adam"
            if self.optimizer_type == NNOptimizer.Ftrl:
                  self.optimizerFunction = "Ftrl"
            if self.optimizer_type == NNOptimizer.Momentum:
                  self.optimizerFunction = "Momentum"
            if self.optimizer_type == NNOptimizer.RMSProp:
                  self.optimizerFunction = "RMSProp"
            return self.optimizerFunction  

      def train(self, data):
            nn = tf.contrib.learn.Estimator(model_fn=self.generateModelFunction, params = self.learning_rate)
            nn.fit(input_fn=self.get_train_inputs, steps=self.num_steps)
            
            ev = nn.evaluate(input_fn=self.getTestInputs, steps=1)
            print("Loss: %s" % ev["loss"])
            print("Root Mean Squared Error: %s" % ev["rmse"])
            return 
 
      def test(self, neural_network):
            # Prediction set is the data we want to predict, so some examples
            test_nn = neural_network
            test_estimation = test_nn.predict(x=self.test_states, as_iterable=True)
            for i, p in enumerate(test_estimation):
                  print("Prediction %s: %s" % (i + 1, p[""]))
      
















