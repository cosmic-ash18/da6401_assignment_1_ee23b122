import numpy as np

# x is a vector, treated as the same below


# d/dx(sigmoid(x)) = sigmoid(x) * (1 - sigmoid(x))
class Sigmoid:

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    
    def backward(self, grad_out):
        return grad_out * self.out * (1 - self.out)

# d/dx tanh(x) = 1 - tanh(x)^2
class Tanh:

    def forward(self, x):
        self.out = np.tanh(x)
        return self.out
    
    def backward(self, grad_out):
        return grad_out * (1 - self.out**2)

# d/dx relu(x) = 1 if x > 0, 0 if x <= 0
class ReLU:

    def forward(self, x):
        self.mask = (x > 0)
        return x *self.mask
    
    def backward(self, grad_out):
        return grad_out * self.mask

# d softmax_i / dx_j = softmax_i * (1 - softmax_i) if i == j
#                       - softmax_i * softmax_j if i != j

# When used with cross-entropy, just becomes dL/d = s - y = grad_out
class Softmax:

    def forward(self, x):
        # Subtract max to maintain nuerical stability
        exp = np.exp(x - np.max(x,axis=1,keepdims=True))
        self.out = exp / np.sum(exp, axis=1, keepdims=True)
        return self.out
    
    def backward(self, grad_out):
        return grad_out
