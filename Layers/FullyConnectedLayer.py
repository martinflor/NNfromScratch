# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:12:04 2022

@author: Florian Martin


"""

import sys
  
sys.path.insert(0, 'C:/Users/Ineed/OneDrive/Bureau/GITHUB/NNfromScratch')

from Layer import Layer
import numpy as np

class Linear(Layer) :
    def __init__(self, input_shape, output_shape, initialization = "uniform") :
        
        self.initialization = initialization
        
        if(self.initialization == "uniform") :
            self.weights = np.random.uniform(low = -1/input_shape, high = 1/input_shape, size = (input_shape, output_shape))
            self.bias    = np.random.uniform(low = -1/input_shape, high = 1/input_shape, size = (input_shape, output_shape))
            
        if(self.initialization == "normal") :
            self.weights = np.random.normal(size = (input_shape, output_shape))
            self.bias    = np.random.normal(size = (input_shape, output_shape))
            
    def forward(self, data_input) :
        self.input  = data_input
        self.output = np.dot(self.weights, self.input) + self.bias
        
        return self.output
    
    def backward(self, output_error, lr, optimizer) :
        """
        
        dE/dx = dE/dy dy/dx => (error at the input) = (error at output) * (weights)
        dE/dB = dE/dy dy/dB = dE/dy
        
        error propagated : dE/dw = dE/dy dy/dw = dEdy x
        
        """
        
        input_error     = np.dot(output_error, self.weights.T)
        weight_error    = np.dot(self.input.T, output_error)
        bias_error      = output_error
        
        
        self.weights = optimizer(self.weights, weight_error, lr)
        self.bias    = optimizer(self.bias, bias_error, lr)
        
        return input_error
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


