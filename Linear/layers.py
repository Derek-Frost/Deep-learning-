import numpy as np
import math
def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    #print(f'predictions = {predictions}')
    predictions -= np.max(predictions, axis = -1, keepdims = True)
    #print(f'predictions = {predictions}')
    sum_exp = 0
    sum_exp += math.e ** predictions
    #print(f'sum_exp = {sum_exp}')
    sum_exp = np.sum(sum_exp, axis=1)
    #print(f'sum_exp after = {sum_exp}, predictions.ndim = {predictions.ndim}')
    probs = np.zeros(predictions.shape)
    for i in range (predictions.shape[0]):
        probs[i] = np.array(math.e ** predictions[i, :])/sum_exp[i]
    #print(f'probs = {probs}')
    return probs

def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    probs = probs[None, :] if probs.ndim < 2 else probs  if isinstance(target_index, np.ndarray) else np.array([target_index])
    idx = target_index #[[],[],[],...]
    mas = np.zeros_like(probs)
    #print(f'mas = {mas}, target_index = {idx}')
    for i in range (probs.shape[0]):
        mas[i,idx[i]] = 1
    #print(f'mas = {mas}, target_index = {idx}')
    probs_lg = np.log(probs)
    #print(f' probs_lg = {probs_lg}, mas * probs_lg = {mas * probs_lg}, -np.sum(mas * probs_lg) = {-np.sum(mas * probs_lg)}')
    loss = -np.sum(mas * probs_lg)/probs.ndim
    #print(f'loss = {loss}')
    return loss


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    sum = np.sum(np.power(W[:,:],2))
    l2_reg_loss = reg_strength * sum
    grad = W * 2
    #print((f'grad = {grad}'))
    return l2_reg_loss, grad


def softmax_with_cross_entropy(preds, target_index):
    """
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
    """
    # TODO: Copy from the previous assignment
    pred = preds.copy()
    num_examples = pred.shape[0]
    exp_scores = np.exp(pred - np.max(pred, axis=1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    
    loss = -np.sum(np.log(probs[np.arange(num_examples), target_index])) / num_examples
    dprediction = probs.copy()
    dprediction[np.arange(num_examples), target_index] -= 1
    dprediction /= num_examples
    #print()
    #print("FINAL")
    #print(f'loss = {loss}, grad = {dprediction}')
    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        
        relu = lambda x: x * (x > 0)
        Y = relu(X)
        self.grad = np.zeros_like(X)
        self.grad += X > 0
        return Y

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        
        d_result = self.grad * d_out 
        
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
       
        self.X = X
        out = np.dot(X, self.W.value) + self.B.value
        return out

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        #print(f'self.W.grad.shape = {self.W.grad.shape}, self.W.value.shape = {self.W.value.shape}, d_out.shape = {d_out.shape}')
        d_B = np.sum(d_out, axis=0, keepdims=True) # Сумма градиента выхода по строкам
        d_result = np.dot(d_out, self.W.value.T)
        d_W = np.dot(self.X.T, d_out)
        #print(f'self.W.grad.shape = {self.W.grad.shape}, d_W.shape = {d_W.shape}, self.X.shape = {self.X.shape}')
        self.W.grad = d_W
        self.B.grad = d_B
        
        return d_result

    def params(self):
        return {'W': self.W, 'B': self.B}
