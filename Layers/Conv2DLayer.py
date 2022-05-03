# -*- coding: utf-8 -*-
"""
Created on Tue May  3 09:16:41 2022

@author: Florian Martin

Convolutionnal Layer

"""

import sys
  
sys.path.insert(0, 'C:/Users/Ineed/OneDrive/Bureau/GITHUB/NNfromScratch')

from Layer import Layer
import numpy as np

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, input_shape = None, kernel_size = (3,3), stride = 1, padding = 0, bias_bool = True):
        """
        in_channels : # of channel's input
        out_channels : # of channel's output
        input_shape : Height and Width of the input
        kernel_size : Filter's size
        
        """
        self.H_in, self.W_in = (None, None)
        if(input_shape is not None) :
            self.H_in, self.W_in = input_shape
            
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size if(isinstance(kernel_size, tuple)) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.bias_bool = bias_bool
        
        return
        
    def init_weight(self) :
        self.weights  = np.random.normal(size = (self.kernel_size[0], self.kernel_size[1], self.in_channels))
        if(self.bias_bool):
            self.bias = np.random.normal(size = (self.kernel_size[0], self.kernel_size[1], self.in_channels))
        
        return (self.weights, self.bias)
    
    def forward(self, data_input) :
        if(self.H_in is None or self.W_in is None) :
            self.H_in, self.W_in = data_input.shape
        
        H_out = 1 + (self.H_in - self.kernel_size[0] + 2*self.padding)//self.stride
        W_out = 1 + (self.W_in - self.kernel_size[1] + 2*self.padding)//self.stride
        
        self.input = data_input
        self.output = np.zeros((H_out, W_out, self.out_channels))
        
        
        for i in range(H_out) :
            for j in range(W_out) :
                H_start = i * self.stride
                H_stop = H_start + self.kernel_size[0]
                
                W_start = j * self.stride
                W_stop = W_start + self.kernel_size[1]
                
                input_part = self.input[H_start:H_stop, W_start:W_stop, self.in_channels]
                
        
        
        
        
        
        
        
        

