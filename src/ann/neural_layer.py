"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np

# One fully connected dense layer of the neural network
class NeuralLayer:

    def __init__(self, in_dim, out_dim, weight_init="xavier"):

        if weight_init == "random":
            self.W = np.random.randn(in_dim, out_dim) * 0.01
        elif weight_init == "xavier":
            limit = np.sqrt(6 / (in_dim + out_dim))
            self.W = np.random.uniform(-limit, limit, (in_dim, out_dim))
        else:
            raise ValueError("Invalid weight initialization")

        self.b = np.zeros((1, out_dim))


    # x : batch_size * in_dim => x.T : in_dim * batch_size
    # W : in_dim * out_dim
    # result : batch_size * out_dim
    def forward(self, x):
        # Store input, needed for backprop
        self.input = x
        return np.dot(x, self.W) + self.b

    # grad_out = dl/dx, the gradient from the next layer : batch_size * out_dim
    def backward(self, grad_out):

        batch_size = self.input.shape[0]

        self.grad_W = np.dot(self.input.T, grad_out)

        # Bias affects every sample equally, so sum gradients
        self.grad_b = np.sum(grad_out, axis=0, keepdims=True)

        # Pass the gradient to the previous layer
        grad_input = np.dot(grad_out, self.W.T)

        return grad_input