# -*- coding: utf-8 -*-
"""
Created on Mon May  2 18:01:12 2022

@author: Florian Martin

Activation Functions

"""


import numpy as np


class ActivationFunction :
    def ReLu():
        return (lambda x : x if x > 0 else 0, lambda x : 1 if x > 0 else 0)
    
    def Sigmoid():
        return (lambda x : 1/(1 + np.exp(-x)), lambda x : np.exp(-x)/(1 + np.exp(-x))**2)
    
    def tanh() :
        return (lambda x : np.tanh(x), lambda x : 1-np.tanh(x)**2)
    
    def PReLu(a) :
        return (lambda x : x if x > 0 else a*x, lambda x : 1 if x > 0 else a)
    
    def softmax() :
        S = lambda z : np.exp(z-np.max(z))/np.sum(np.exp(z-np.max(z)))
        return (lambda x : S(x), lambda x : -np.outer(S(x), S(x)) + np.diag(S(x).flatten()))
