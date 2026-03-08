"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
import wandb
# Import one fully connected layer
from ann.neural_layer import NeuralLayer
# Import activations 
from ann.activations import ReLU, Sigmoid, Tanh
# Import Loss functions
from ann.objective_functions import CrossEntropy, MSE
from ann.optimizers import SGD, Momentum, NAG, RMSProp


# class of the entire Neural Network model
class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    
    Args:
        cli_args: Command-line arguments for configuring the network
    """

    def __init__(self, cli_args):

        self.cli_args = cli_args
        self.layers = []
        self.activations = []

        # 28*28 image = 784 pixels
        input_dim = 784

        # extract number of hidden layers, activation name and weights 
        # from command line args
        hidden_layers = cli_args.hidden_size
        activation_name = cli_args.activation
        weight_init = cli_args.weight_init

        dims = [input_dim] + hidden_layers + [10] # becomes [784, 128, 64, 10]

        # Create layers (append to self.layers)
        for i in range(len(dims) - 1):

            self.layers.append(
                NeuralLayer(dims[i], dims[i+1], weight_init)
            )

            # Activations for hidden layers only
            if i < len(dims) - 2:
                self.activations.append(self._get_activation(activation_name))

        # get loss function from the command line
        self.loss_fn = self._get_loss(cli_args.loss)

        # get optimizer the similar way
        self.optimizer = self._get_optimizer(cli_args)

    # get function from string
    def _get_activation(self, name):

        if name == "relu":
            return ReLU()
        if name == "sigmoid":
            return Sigmoid()
        if name == "tanh":
            return Tanh()

        raise ValueError("Invalid activation")

    # get loss function from the string
    def _get_loss(self, name):

        if name == "cross_entropy":
            return CrossEntropy()
        if name == "mse":
            return MSE()

        raise ValueError("Invalid loss")

    # Get optimizer function from the string
    def _get_optimizer(self, args):

        if args.optimizer == "sgd":
            return SGD(args.learning_rate)

        if args.optimizer == "momentum":
            return Momentum(args.learning_rate)

        if args.optimizer == "nag":
            return NAG(args.learning_rate)

        if args.optimizer == "rmsprop":
            return RMSProp(args.learning_rate)

        raise ValueError("Invalid optimizer")

    def forward(self, X):
        """
        Forward propagation through all layers.
        Returns logits (no softmax applied)
        X is shape (b, D_in) and output is shape (b, D_out).
        b is batch size, D_in is input dimension, D_out is output dimension.
        """

        out = X

        # hidden layers
        for layer, act in zip(self.layers[:-1], self.activations):
            out = layer.forward(out)
            out = act.forward(out)

        # final linear layer (returns logits as mentioned in the assignment)
        out = self.layers[-1].forward(out)

        return out
    

    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        Returns two numpy arrays: grad_Ws, grad_bs.
        - `grad_Ws[0]` is gradient for the last (output) layer weights,
          `grad_bs[0]` is gradient for the last layer biases, and so on.
        """

        # Compute dL/dy_pred
        grad = self.loss_fn.backward(y_pred, y_true)

        # last layer
        grad = self.layers[-1].backward(grad)

        # hidden layers, need to go in reverse order
        for layer, act in reversed(list(zip(self.layers[:-1], self.activations))):

            grad = act.backward(grad)
            grad = layer.backward(grad)

        grad_w = [layer.grad_W for layer in reversed(self.layers)]
        grad_b = [layer.grad_b for layer in reversed(self.layers)]

        return grad_w, grad_b
    

    # Function which calls the optimizer
    def update_weights(self):
        """
        Update weights using the optimizer.
        """

        self.optimizer.step(self.layers)


    # mini-batch GD
    def train(self, X_train, y_train, epochs, batch_size):
        """
        Train the network for specified epochs.
        """

        n = X_train.shape[0] # size of dataset

        for epoch in range(epochs):

            perm = np.random.permutation(n) # randomly shuffle the dataset
            X_train = X_train[perm]
            y_train = y_train[perm]

            epoch_loss = 0.0
            num_batches = 0

            # for each mini_batch
            for i in range(0, n, batch_size):

                # get the data in this mini batch range
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                
                # predicted y is just a forward pass thorugh the network
                y_pred = self.forward(X_batch)

                # Get the loss (use return value)
                batch_loss = self.loss_fn.forward(y_pred, y_batch)
                epoch_loss += float(batch_loss)
                num_batches += 1

                # do a backprop to get gradients wrt all weights and biases
                self.backward(y_batch, y_pred)
                
                # for 2.4
                if hasattr(self.cli_args, "wandb_project") and self.cli_args.wandb_project is not None:
                    grad_norm = np.mean(np.abs(self.layers[0].grad_W))
                    wandb.log({"grad_norm_layer1" : grad_norm})
                # update the weights
                self.update_weights()

            epoch_loss /= max(1, num_batches)
            #print(f"Epoch {epoch+1}/{epochs} — avg batch loss: {epoch_loss:.6f}")


    # evaluate the final result
    def evaluate(self, X, y):
        """
        Evaluate the network on given data.
        """

        logits = self.forward(X) # logits are the output

        # loss computation
        loss = self.loss_fn.forward(logits, y)

        # predictions are the one hot encoded (so take argmax)
        preds = np.argmax(logits, axis=1)

        # similarly for labels
        labels = np.argmax(y, axis=1)

        # accuracy is just number of equals
        accuracy = np.mean(preds == labels)

        return logits, loss, accuracy


    # function that just makes and returns a dictionary
    # of the weights and biases of each layer
    def get_weights(self):

        weights = {}

        for i, layer in enumerate(self.layers):

            weights[f"W{i}"] = layer.W
            weights[f"b{i}"] = layer.b

        return weights


    # given a weights dictionary, set them as those of our model
    # used in inference
    def set_weights(self, weights):
    # ensure we got a dict and copy arrays into layers
        if weights is None or not isinstance(weights, dict):
            raise ValueError("set_weights expects a dict of numpy arrays.")

        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key not in weights or b_key not in weights:
                raise KeyError(f"Missing keys in weights dict: expected {w_key} and {b_key}")
            # use .copy() to avoid any weird aliasing / pickling side effects
            layer.W = weights[w_key].copy()
            layer.b = weights[b_key].copy()