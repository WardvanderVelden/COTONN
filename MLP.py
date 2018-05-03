# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 19:57:18 2018

@author: Rob Neelen
"""
from __future__ import print_function
import tensorflow as tf
# import matplotlib.pyplot as plt
# import numpy as np

class MLP:
      def __init__(self):
            self.learning_rate = 0.1
            self.num_steps = 100
            self.batch_size = 128
            self.display_step = 100
            self.n_layer = [256, 256, 256, 50]
            self.num_input = 784
            self.num_classes = 10
     
      def setLearningRate(self, value): self.learning_rate = value
      def setNumSteps(self, value): self.num_steps = value
      def setBatchSize(self, value): self.batch_size = value
      def setDisplayStep(self, value): self.display_step = value
      def setNeurons(self, value): self.n_layer = value
      def setNumInput(self, value): self.num_input = value
      def setNumClasses(self, value): self.num_classes = value
      
      def getLearningRate(self): return self.learning_rate
      def getNumSteps(self, value): return self.num_steps
      def getBatchSize(self, value): return self.batch_size
      def getDisplayStep(self, value): return self.display_step
      def getNeurons(self, value): return self.n_layer 
      def getNumInput(self, value): return self.num_input
      def getNumClasses(self, value): return self.num_classes
      
      
      #Define the input function for training
      def setInputFunction(self, data):
            input_features = {'images': data.train.images}
            input_labels = data.train.labels
      
            #Define the input function for training
            self.input_fn = tf.estimator.inputs.numpy_input_fn(
                        input_features, 
                        input_labels,
                        batch_size=self.batch_size, 
                        num_epochs=None, 
                        shuffle=True)
            return self.input_fn
            
      
      #Define the Neural Network
      def neural_net(self, input_features_dict):
            """
            In neural_net the structure of the Neural Network is created. 
            The network consists of 1 input layer, 1 output layer and a variable
             amount of hidden layers. 
            The input layer is created. 
            Followed by the hidden layers, who are first created as a empty list,
             than filled up by the number of hidden layers we wish to build. 
            At last the output layer is build with the last hidden layer as input. 
            """
            # TF Estimator input is a dict, in case of multiple inputs
            input_features = input_features_dict['images']         
            # Build Input Layer
            in_layer = tf.layers.dense(input_features, self.n_layer[0])
            # Build Hidden layers, based on the amount of hidden layers required
            hidden_layers = []
            hidden_layers[0] = tf.layers.dense(in_layer, self.n_layer[0])
            for i in range(0, len(self.n_layer)-1):
                  hidden_layers[i] =  tf.layers.dense(hidden_layers[i-1], self.n_layer[i])       
            # Build the output layer with the last hidden layer as input
            out_layer = tf.layers.dense(hidden_layers[-1], self.num_classes)
            return out_layer
      
      
      def model_fn(self, features, labels, mode):
            # Build the neural network
            logits = self.neural_net(features)
            
            # Predictions
            pred_classes = tf.argmax(logits, axis=1)
            # pred_probas = tf.nn.softmax(logits)
      
            # If prediction mode, early return
            if mode == tf.estimator.ModeKeys.PREDICT:
                  return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)
      
            # Define Loss and Optimizer
            loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                  logits=logits, 
                  labels=tf.cast(labels, dtype=tf.int32)))
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())
      
            # Evaluate the accuracy of the model
            acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
      
            # TF Estimators requires to return a EstimatorSpec, that specify
            # the different ops for training, evaluation ...
            estim_specs = tf.estimator.EstimatorSpec(
                  mode=mode,
                  predictions=pred_classes,
                  loss=loss_op,
                  train_op=train_op,
                  eval_metric_ops={'accuracy': acc_op})
            return estim_specs
      
      def buildNN(self, data): 
                  model = tf.estimator.Estimator(self.model_fn)
                  model.train(self.setInputFunction(data))
                  NN_evaluation = model.evaluate(self.setInputFunction(data))
                  return NN_evaluation
                  
                  
            

      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      