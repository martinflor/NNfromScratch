# -*- coding: utf-8 -*-
"""
Created on Mon May  2 19:18:34 2022

@author: Florian Martin

Loss Function

"""





import numpy as np


class LossFunction :
    def MSE():
        return (lambda y_true, y_hat : np.mean(np.power(y_true - y_hat), 2), lambda y_true, y_hat : 2*(y_hat-y_true)/y_true.size)

    def cross_entropy():
        return (lambda y_true, y_hat : -y_true*np.log(y_hat) -(1-y_true)*np.log(1-y_hat), lambda y_true, y_hat : (y_hat-y_true)/((1-y_hat)*y_hat))