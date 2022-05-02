# -*- coding: utf-8 -*-
"""
Created on Mon May  2 17:42:00 2022

@author: Florian Martin

Activation Layer

"""


import sys
  
sys.path.insert(0, 'C:/Users/Ineed/OneDrive/Bureau/GITHUB/NNfromScratch')

from Layer import Layer

class Activation(Layer) :
    def __init__(self, func, dfunc):
        self.func, self.dfunc = func, dfunc
        
    def forward(self, data_input) :
        self.input = data_input
        self.output = self.output(self.input)
        
        return self.output
    
    def backward(self, output_error, lr) :
        """
        dE/dx = dE/dy dy/dx = dE/dy * dfunc(x) (ELEMENT-WISE)
        """
        input_error = output_error * self.dfunc(self.input)
        
        return input_error
        

