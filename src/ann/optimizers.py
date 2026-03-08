"""
Optimization Algorithms
Implements: SGD, Momentum, NAG, RMSProp, Adam, Nadam
"""

# Simple implementations
#

import numpy as np


# SGD
class SGD:

    def __init__(self, lr):
        self.lr = lr

    def step(self, layers):

        for layer in layers:

            # basic gradient descent
            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b


# Momentum GD
class Momentum:

    def __init__(self, lr, beta=0.9):

        self.lr = lr
        self.beta = beta
        self.vW = None
        self.vb = None

    def step(self, layers):

        # start the velocity term first time
        if self.vW is None:

            self.vW = []
            self.vb = []

            for layer in layers:
                self.vW.append(np.zeros_like(layer.W))
                self.vb.append(np.zeros_like(layer.b))



        for i, layer in enumerate(layers):

            # velocit update here
            self.vW[i] = self.beta * self.vW[i] + layer.grad_W
            self.vb[i] = self.beta * self.vb[i] + layer.grad_b

            # W&B update here
            layer.W -= self.lr * self.vW[i]
            layer.b -= self.lr * self.vb[i]



# NAG is Nesterov accelerated gradient
class NAG:
    # beta is the momentum coefficient
    def __init__(self, lr, beta=0.9):
        # store params
        self.lr = lr
        self.beta = beta

        # velocity
        self.vW = None
        self.vb = None

    # update all layer weights
    def step(self, layers):

        # only do first time optimizer runs
        if self.vW is None:

            # store velocity per layer
            self.vW = []
            self.vb = []

            for layer in layers:
                self.vW.append(np.zeros_like(layer.W))
                self.vb.append(np.zeros_like(layer.b))

        # go over all layers
        for i, layer in enumerate(layers):

            # storing the old velocity for nesterov lookahead
            prev_vW = self.vW[i]
            prev_vb = self.vb[i]

            # velocity is updated here
            self.vW[i] = self.beta * self.vW[i] + self.lr * layer.grad_W
            self.vb[i] = self.beta * self.vb[i] + self.lr * layer.grad_b

            # Nesterov update
            layer.W -= (self.beta * prev_vW + (1 - self.beta) * self.vW[i])
            layer.b -= (self.beta * prev_vb + (1 - self.beta) * self.vb[i])



# Scale learning rate using running avg of squared gradients
class RMSProp:
    # beta is decay rate, eps for numeric stability
    # (eps prevents division by 0)
    def __init__(self, lr, beta=0.9, eps=1e-8):

        self.lr = lr
        self.beta = beta
        self.eps = eps

        # square gradient accumulatros
        self.sW = None
        self.sb = None

    # update params here
    def step(self, layers):

        if self.sW is None:

            self.sW = []
            self.sb = []

            # create zero arrays for all layers
            for layer in layers:
                self.sW.append(np.zeros_like(layer.W))
                self.sb.append(np.zeros_like(layer.b))

        for i, layer in enumerate(layers):

            # update running average of squared gradients
            self.sW[i] = self.beta * self.sW[i] + (1 - self.beta) * (layer.grad_W ** 2)
            self.sb[i] = self.beta * self.sb[i] + (1 - self.beta) * (layer.grad_b ** 2)

            # update the actual params here
            layer.W -= self.lr * layer.grad_W / (np.sqrt(self.sW[i]) + self.eps)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(self.sb[i]) + self.eps)