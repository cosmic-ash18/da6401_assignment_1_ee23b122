"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np

# cross entropy loss class
# recall its for multiclass classification (which MNIST is)
class CrossEntropy:

    def forward(self, logits, y_true):
        # logits: (N, C)
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.pred = exp / np.sum(exp, axis=1, keepdims=True)

        # allow y_true to be either integer labels (N,) or one-hot (N, C)
        if y_true.ndim == 1:
            # convert class indices to one-hot
            C = self.pred.shape[1]
            y_one = np.zeros((y_true.shape[0], C))
            y_one[np.arange(y_true.shape[0]), y_true] = 1
            self.y_true = y_one
        else:
            self.y_true = y_true

        loss = -np.sum(self.y_true * np.log(self.pred + 1e-9)) / self.y_true.shape[0]
        return loss

    def backward(self, logits=None, y_true=None):
        # If logits and labels provided externally (autograder may call like this)
        if logits is not None and y_true is not None:
            exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            pred = exp / np.sum(exp, axis=1, keepdims=True)

            # convert integer labels -> one-hot if needed
            if y_true.ndim == 1:
                C = pred.shape[1]
                y_one = np.zeros((y_true.shape[0], C))
                y_one[np.arange(y_true.shape[0]), y_true] = 1
                y_true_proc = y_one
            else:
                y_true_proc = y_true

            # return averaged gradient (dL/dz = (pred - y)/N)
            return (pred - y_true_proc) / y_true_proc.shape[0]

        # normal backward after forward() — self.pred and self.y_true available
        return (self.pred - self.y_true) / self.y_true.shape[0]


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
    def backward(self, pred=None, target=None):

        # autograder-style call
        if pred is not None and target is not None:
            n = pred.shape[0]
            return 2 * (pred - target) / n

        # normal backward after forward()
        n = self.pred.shape[0]
        return 2 * (self.pred - self.target) / n