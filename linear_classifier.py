import torch
import random


class LinearClassifier:
    def __init__(self, seed=42):
        """
        Initialize the Linear class
        """
        random.seed(seed)
        torch.manual_seed(seed)
        self.W = None

    def __get_batch(self, X, y, number_training, batch_size):
        indices = torch.randint(0, number_training, (batch_size,))
        X_batch = X[indices, :]
        y_batch = y[indices]
        return X_batch, y_batch

    def __svm(W, X, y, reg):
        loss = 0.0

        # initialize the gradient as zero
        dW = torch.zeros_like(W)
        number_training = X.shape[0]

        # Calculating Loss
        scores = X.mm(W)
        correct_class_scores = scores[range(number_training), y]
        margin = scores - correct_class_scores.view(-1, 1) + 1
        margin[margin < 0] = 0
        margin[range(number_training), y] = 0
        loss = margin.sum()
        loss /= number_training
        loss += reg * torch.sum(W * W)

        # Calculating Gratident of SVM
        binary = margin.clone()
        binary[margin > 0] = 1
        row_sum = binary.sum(dim=1)
        binary[range(number_training), y] = -1 * row_sum
        dW = torch.mm(X.t(), binary)
        dW /= number_training
        dW += 2 * reg * W

        return loss, dW

    def train(self, X: torch.Tensor, y: torch.Tensor, lr: float = 1e-2, reg: float = 1e-2, iter_count=100, batch_size=100):
        # Initialize Everything
        number_training = X.shape[0]
        image_dimention = X.shape[1]
        number_classes = torch.max(y)
        W = 0.000001 * torch.randn(
            image_dimention, number_classes, device=X.device, dtype=X.dtype
        )
        loss_history = []

        # Main Iteration for training

        for _ in range(iter_count):
            X_batch, y_batch = self.__get_batch(
                X, y, number_training, batch_size)

            loss, gradient = self.__svm(W, X_batch, y_batch, reg)
            loss_history.append(loss.item())

            # Gradient upate to weights
            W -= lr * gradient

        self.W = W
        return loss_history
    
    def predict(self, X: torch.Tensor):
        # Initialized
        y_pred = torch.zeros(X.shape[0], dtype=torch.int64)
        y_pred = X.mm(self.W).argmax(dim=1)

        return y_pred
