import numpy as np
import math
from numpy import diff


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    pred = predictions.copy()
    
    probs = softmax(pred)
    #print("probs", probs)
    loss = cross_entropy_loss(probs, target_index)
    indicator = np.zeros_like(probs)
    idx = target_index
    for i in range (probs.shape[0]):
        indicator[i,idx[i,0]] = 1
    #print("indicator", indicator)
    dprediction = (probs - indicator) / probs.ndim
    #print()
    #print("FINAL")
    #print(f'loss = {loss}, grad = {dprediction}')
    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
      l2_reg_loss = regularization_strength * sumij W[i, j]2
    '''
    #print(f'W = {W}')
    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops 
    sum = np.sum(np.power(W[:,:],2))
    #print(f'sum = {sum}')
    l2_reg_loss = reg_strength * sum
    grad = W * 2
    #print((f'grad = {grad}'))
    return l2_reg_loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)
    exp_scores = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    num_examples = X.shape[0]
    loss = -np.sum(np.log(probs[np.arange(num_examples), target_index])) / num_examples

    dscores = probs.copy()
    dscores[np.arange(num_examples), target_index] -= 1
    dscores /= num_examples
    dW = np.dot(X.T, dscores)
    #print(f'X = {X}, dscores = {dscores}, dW = {dW}')
    #print(f'X.shape = {X.shape}, dscores.shape = {dscores.shape}, dW.shape = {dW.shape}')
    #print(f'predictions = {predictions}, target_index = {target_index}')
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None
        
    

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            idx_batch = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[idx_batch]
            y_batch = y[idx_batch]
            loss, grad = linear_softmax(X_batch, self.W, y_batch)
            loss_W, grad_W = l2_regularization(self.W, reg)
            self.W += -learning_rate * (grad_W + grad)
            # end
            #print("Epoch %i, loss: %f" % (epoch, loss_W))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        y_pred = np.zeros(X.shape[0])
        scores = np.dot(X, self.W)
        y_pred = np.argmax(scores, axis=1)
        return y_pred



                
                                                          

            

                
