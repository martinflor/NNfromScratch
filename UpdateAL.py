# -*- coding: utf-8 -*-
"""
Created on Mon May  2 20:11:31 2022

@author: Florian Martin

Updating algorithms

"""



def GradD() :
    return lambda weight, error, lr : weight - lr*error
    

