# -*- coding: utf-8 -*-
"""
Created on Mon May  2 19:30:48 2022

@author: Florian Martin


Neural Network from scratch

"""

import sys
  
sys.path.insert(0, 'C:/Users/Ineed/OneDrive/Bureau/GITHUB/NNfromScratch/Layers')
sys.path.insert(0, 'C:/Users/Ineed/OneDrive/Bureau/GITHUB/NNfromScratch/Functions')  
sys.path.insert(0, 'C:/Users/Ineed/OneDrive/Bureau/GITHUB/NNfromScratch')

from FullyConnectedLayer import Linear
from UpdateAL import GradD


class Neural:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.dloss = None
        
    def add(self, layer):
        self.layers.append(layer)
        
    def set_loss(self, loss, dloss):
        self.loss = loss
        self.dloss = dloss
    
    def predict(self, data_input):
        data_output = []
        
        for data in data_input :
            for layer in self.layers :
                data = layer.forward(data)
                
            data_output.append(data)
            
        return data_output
    
    def fit(self, x, y, lr=0.01, epochs=100, optimizer = GradD):
        """
        x, y : training examples
        """
        size = len(x)
        
        for i in range(epochs) : # For each epochs 
            forward_error = 0.0
            for j in range(size) : # For each training example
                data = x[j]
                for layer in self.layers :
                    data = layer.forward(data)
                
                forward_error += self.loss(y[j], data)
                
                error = self.dloss(y[j], data) # dE/dy for the last layer : loss function
                for layer in reversed(self.layers) :
                    if(isinstance(layer, Linear)) :
                        error += layer.backward(error, lr, optimizer)
                    else :
                        error += layer.backward(error, lr)
                    
            forward_error /= size
            print('Epoch %d/%d   error=%f' % (i+1, epochs, forward_error))
                
                
            
            
            
            
            

