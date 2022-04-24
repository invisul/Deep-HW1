import torch
from torch import Tensor
from collections import namedtuple
from torch.utils.data import DataLoader

from .losses import ClassifierLoss


class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        w_shp = (self.n_features, self.n_classes)  # shape of the weight matrix
        self.weights = torch.normal(mean=torch.zeros(w_shp), std=weight_std*torch.ones(w_shp))

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """
        class_scores = x @ self.weights
        y_pred = torch.argmax(class_scores, dim=1)

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """
        assert y.shape == y_pred.shape
        assert y.dim() == 1
        acc = torch.sum(y == y_pred).item() / len(y)
        return acc * 100

    def train(
        self,
        dl_train: DataLoader,
        dl_valid: DataLoader,
        loss_fn: ClassifierLoss,
        learn_rate=0.1,
        weight_decay=0.001,
        max_epochs=100,
    ):

        Result = namedtuple("Result", "accuracy loss")
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print("Training", end="")
        for epoch_idx in range(max_epochs):
            # total_correct = 0
            # average_loss = 0

            #  Implement model training loop.
            #  1. At each epoch, evaluate the model on the entire training set
            #     (batch by batch) and update the weights.
            #  2. Each epoch, also evaluate on the validation set.
            #  3. Accumulate average loss and total accuracy for both sets.
            #     The train/valid_res variables should hold the average loss
            #     and accuracy per epoch.
            #  4. Don't forget to add a regularization term to the loss,
            #     using the weight_decay parameter.

            # 1) evaluate on training set batch by batch, and update weights for each batch
            total_loss = 0
            total_accuracy = 0
            num_iter = 0
            for idx, (x, y) in enumerate(dl_train):
                num_iter += 1
                y_pred, x_scores = self.predict(x)
                loss = loss_fn.loss(x, y, x_scores, y_pred)
                grad = loss_fn.grad()
                total_loss += loss
                total_accuracy += self.evaluate_accuracy(y, y_pred)
                self.weights -= learn_rate * (grad * loss + weight_decay * self.weights)

            train_res.accuracy.append(total_accuracy / num_iter)
            train_res.loss.append(total_loss / num_iter)

            total_loss = 0
            total_accuracy = 0
            num_iter = 0
            for idx, (x, y) in enumerate(dl_valid):
                num_iter += 1
                y_pred, x_scores = self.predict(x)
                loss = loss_fn.loss(x, y, x_scores, y_pred)
                total_loss += loss
                total_accuracy += self.evaluate_accuracy(y, y_pred)

            valid_res.accuracy.append(total_accuracy / num_iter)
            valid_res.loss.append(total_loss / num_iter)

            print(".", end="")

        print("")
        return train_res, valid_res

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).
        tmp = 0
        if has_bias:
            tmp = self.weights[1:, :]
        else:
            tmp = self.weights
        w_images = tmp.T.view(-1, *img_shape)

        return w_images


def hyperparams():
    hp = dict(weight_std=0.02, learn_rate=0.02, weight_decay=0.02)

    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.

    return hp
