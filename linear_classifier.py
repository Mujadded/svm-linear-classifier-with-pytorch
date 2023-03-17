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

    def __svm(self, W, X, y, reg):
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

    def calculate_accuracy(self, y_true, y_pred):
        number_training = y_pred.shape[0]
        acc = ((number_training - (y_pred -
                        y_true).count_nonzero())/number_training)
        return acc.item()

    def train(self, X: torch.Tensor, y: torch.Tensor, X_val: torch.Tensor, y_val: torch.Tensor, lr: float = 1e-2, reg: float = 1e-2, num_iterations: int = 100, batch_size: int = 100):
        # Initialize Everything
        number_training = X.shape[0]
        number_validations = X_val.shape[0]
        image_dimention = X.shape[1]
        number_classes = torch.max(y)+1
        self.W = 0.000001 * torch.randn(
            image_dimention, number_classes, device=X.device, dtype=X.dtype
        )
        history = {'loss': [], 'train_acc': 0, 'val_acc': 0}

        # Main Iteration for training

        for num_iteration in range(num_iterations):
            X_batch, y_batch = self.__get_batch(
                X, y, number_training, batch_size)

            loss, gradient = self.__svm(self.W, X_batch, y_batch, reg)
            history['loss'].append(loss.item())

            # Gradient upate to weights
            self.W -= lr * gradient      
        
        y_pred_train = self.predict(X)
        train_acc = self.calculate_accuracy(y,y_pred_train)
        history['train_acc'] = train_acc

        y_pred_validation = self.predict(X_val)
        val_acc = self.calculate_accuracy(y_val,y_pred_validation)
        history['val_acc'] = val_acc

        print(f'Trainning Accuracy: {train_acc:.2f}, Validation Accuracy: {val_acc:.2f}')

        return history

    def predict(self, X: torch.Tensor):
        # Initialized
        y_pred = torch.zeros(X.shape[0], dtype=torch.int64)
        W = self.W
        y_pred = X.mm(W).argmax(dim=1)

        return y_pred
        
