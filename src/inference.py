"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import numpy as np
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


def parse_arguments():
    """
    Parse command-line arguments for inference.
    """

    # Same as the parse arguments in train.py
    parser = argparse.ArgumentParser(description='Run inference on test set')

    parser.add_argument('-d','--dataset', type=str, default="mnist")
    parser.add_argument('-e','--epochs', type=int, default=10)
    parser.add_argument('-b','--batch_size', type=int, default=64)
    parser.add_argument('-l','--loss', type=str, default='cross_entropy')
    parser.add_argument('-o','--optimizer', type=str, default='rmsprop')
    parser.add_argument('-lr','--learning_rate', type=float, default=0.001)
    parser.add_argument('-wd','--weight_decay', type=float, default=0.0)
    parser.add_argument('-nhl','--num_layers', type=int, default=1)
    parser.add_argument('-sz','--hidden_size', type=int, nargs='+', default=[128, 64])
    parser.add_argument('-a','--activation', type=str, default='relu')
    parser.add_argument('-w_i','--weight_init', type=str, default='xavier')
    parser.add_argument('-w_p','--wandb_project', type=str, default=None)

    parser.add_argument('--model_path', type=str, default='best_model.npy')

    return parser.parse_args()


# load the model
def load_model(model_path):
    """
    Load trained model from disk.
    """
    # np.load loadas a numpy object, allow_pickle = True allows
    # pythons objects, .item()
    data = np.load(model_path, allow_pickle=True).item()
    return data # return the loaded model


# very similar to the evaluate function in neural_network.py
def evaluate_model(model, X_test, y_test):

    logits = model.forward(X_test)

    preds = np.argmax(logits, axis=1)
    labels = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(labels, preds)

    # precision = true positive / true positive + false positive
    # macro averages it over all classes
    precision = precision_score(labels, preds, average="macro",zero_division=0)

    # recall = true positive / true positive + false negative
    recall = recall_score(labels, preds, average="macro",zero_division=0)
    f1 = f1_score(labels, preds, average="macro",zero_division=0)

    # return this evaluation benchmarks
    return {
        "logits": logits,
        "loss": None,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "preds": preds,
        "labels": labels
    }


# The main functon putting it all together
def main():

    # parse the arguments
    args = parse_arguments()

    # Same as in train.py
    if args.wandb_project is not None:
        wandb.init(project=args.wandb_project, config=vars(args))

    # load the data
    _, _, _, _, X_test, y_test = load_dataset(args.dataset)

    # Get the model with the passed arguments
    model = NeuralNetwork(args)

    # load the trained weights (already have this cuz we run train.py)
    weights = np.load(args.model_path, allow_pickle=True).item()

    print("Loaded weight keys:", sorted(weights.keys()))
    model.set_weights(weights)

    # evaluate the model
    results = evaluate_model(model, X_test, y_test)

    preds = results["preds"]
    labels = results["labels"]

    # compute confusion matrix
    cm = confusion_matrix(labels, preds)

    # class labels (same for MNIST and Fashion-MNIST: 0–9)
    class_names = [str(i) for i in range(10)]

    if args.wandb_project is not None:

        wandb.log({
            "test_accuracy" : results["accuracy"],
            "test_f1" : results["f1"],
            "test_precision" : results["precision"],
            "test_recall" : results["recall"]
        })

        # log confusion matrix to wandb
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                preds=preds,
                y_true=labels,
                class_names=class_names
            )
        })

    print("Evaluation complete!")
    print(results)  # print the evaluated benchmarks

    if args.wandb_project is not None:
        wandb.finish()

    return results


# entry point
if __name__ == "__main__":
    main()