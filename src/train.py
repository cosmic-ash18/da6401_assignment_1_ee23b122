"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import numpy as np
import json
import wandb
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset
from sklearn.metrics import f1_score, confusion_matrix

# Function which parses the arguments passed in the command line
def parse_arguments():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(description='Train a neural network')

    # passing -d and --dataset means allow both of them to be used, -d is the shorthand
    # required=True means something must be passed, else it fallsback to default 
    parser.add_argument('-d','--dataset', type=str, default="mnist")
    parser.add_argument('-e','--epochs', type=int, default=20)
    parser.add_argument('-b','--batch_size', type=int, default=128)
    parser.add_argument('-l','--loss', type=str, default='cross_entropy')
    parser.add_argument('-o','--optimizer', type=str, default='rmsprop')
    parser.add_argument('-lr','--learning_rate', type=float, default=0.0002)
    parser.add_argument('-wd','--weight_decay', type=float, default=0.0)
    parser.add_argument('-nhl','--num_layers', type=int, default=2)
    parser.add_argument('-sz','--hidden_size', type=int, nargs='+', default=[256, 128])
    parser.add_argument('-a','--activation', type=str, default='relu')
    parser.add_argument('-w_i','--weight_init', type=str, default='xavier')
    parser.add_argument('-w_p','--wandb_project', type=str, default=None)


    # return the parsed arguments
    return parser.parse_args()


# Putting it all together
def main():
    """
    Main training function.
    """

    # Parse the arguments passed by me
    args = parse_arguments()

    # fix for wandb sweep sending hidden_size as int
    if isinstance(args.hidden_size, int):
        args.hidden_size = [args.hidden_size]

    # initialize wandb only if project name is provided
    if args.wandb_project is not None:
        wandb.init(project=args.wandb_project,config=vars(args))
    # Get all the datasets using the load_dataset function made in utils/data_loader.py
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(args.dataset)


    # For 2.1 - data exploration
    if args.wandb_project is not None:
        table = wandb.Table(columns=["image", "label"])

        for i in range(50): # around 5 images per class
            img = X_train[i].reshape(28, 28)
            label = np.argmax(y_train[i])

            table.add_data(wandb.Image(img), label)
        
        wandb.log({"sample_images" : table})

    # Get the custom model from the passed arguments
    model = NeuralNetwork(args)

    best_f1 = 0 # store the best f1 score and weights
    best_weights = None 

    # loop through the entire dataset args.epochs number of times
    for epoch in range(args.epochs):
        # 1 here is just number of epochs (check out train in neural_network.py)
        model.train(X_train, y_train, 1, args.batch_size)

        # Get the logits, loss and accuracy (returnees of evaluate function in neural_network.py)
        logits, loss, acc = model.evaluate(X_val, y_val)

        # predictions and labels actual from one-hot encoding
        preds = np.argmax(logits, axis=1)
        labels = np.argmax(y_val, axis=1)
        
        # get the f1 score
        # zero_division=0 removes the sklearn warnings
        f1 = f1_score(labels, preds, average="macro", zero_division=0)


        if args.wandb_project is not None:
            wandb.log({"epoch" : epoch, "val_loss" : loss,
                    "val_accuracy" : acc, "val_f1" : f1})
        # Store only the best f1 score
        if f1 > best_f1:
            best_f1 = f1
            # the corresponding weights are the best configuration
            best_weights = model.get_weights()

    if best_weights is None:
        best_weights = model.get_weights()
    # save the model weights using np.save in the file called best_model.npy
    np.save("best_model.npy", best_weights)

    # restore best model
    model.set_weights(best_weights)
    # Stor the corresponding hyperparams which give the best model (f1 score)
    # args.dataset = "mnist", args.batch_size = 64, ...
    # this is converted to a dictionary using vars()
    # open as a write file
    with open("src/best_config.json","w") as f:
        json.dump(vars(args), f)
    
    logits_test, _, _ = model.evaluate(X_test, y_test)

    preds = np.argmax(logits_test, axis=1)
    labels = np.argmax(y_test, axis=1)

    if args.wandb_project is not None:
        wandb.log({"confusion matrix" : wandb.plot.confusion_matrix(probs=None, y_true=labels, preds=preds)})

    print("Training complete!")

    if args.wandb_project is not None:
        wandb.finish()

# entry point
if __name__ == '__main__':
    main()