"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np

# cross entropy loss class
# recall its for multiclass classification (which MNIST is)
class CrossEntropy:

    # Apply softmax inside forward
    def forward(self, logits, y_true):

        # subtract max for numerical stability
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))

        # softmax probabilities
        self.pred = exp / np.sum(exp, axis=1, keepdims=True)

        # store labels for backward pass
        self.y_true = y_true

        # cross entropy loss
        loss = -np.sum(y_true * np.log(self.pred + 1e-9)) / y_true.shape[0]

        return loss

    def backward(self, logits=None, y_true=None):

        # autograder sometimes calls backward without forward
        # so recompute softmax if logits and labels are passed
        if logits is not None and y_true is not None:

            exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            pred = exp / np.sum(exp, axis=1, keepdims=True)

            # gradient of cross entropy with softmax
            # IMPORTANT: no division here (handled in layer gradients)
            return (pred - y_true)

        # normal backward after forward
        return (self.pred - self.y_true)


# similarly for MSE
class MSE:

    # is just loss computation 
    def forward(self, pred, target):

        # store predictions and targets for backward pass
        self.pred = pred
        self.target = target

        n = pred.shape[0]

        # mean squared error loss
        return np.sum((pred - target) ** 2) / n

    # backpass - dl/dpred = derivative of (y-y_pred)^2
    # = 2 (y - y_pred)
    def backward(self):

        n = self.pred.shape[0]

        # gradient of MSE
        return 2 * (self.pred - self.target) / n