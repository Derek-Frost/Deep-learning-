import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.fc1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu1 = ReLULayer()
        self.fc2 = FullyConnectedLayer(hidden_layer_size, n_output)
        self.relu2 = ReLULayer()
        
    

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        
        params = self.params()
        for param_name in params:
            params[param_name].grad = np.zeros_like(params[param_name].value)  # Инициализация градиентов параметров нулями
        
        # Forward pass
        out = self.fc1.forward(X)  # Прямой проход через первый полносвязный слой
        out = self.relu1.forward(out)  # Применение слоя активации ReLU
        scores = self.fc2.forward(out)  # Прямой проход через второй полносвязный слой
        
        #Compute loss and fill param gradients
        loss, d_scores = softmax_with_cross_entropy(scores, y)  # Вычисление функции потерь и градиента потерь по выходам
        #Backward pass
        d_out = self.fc2.backward(d_scores)  # Обратный проход через второй полносвязный слой
        d_out = self.relu1.backward(d_out)  # Обратный проход через слой активации ReLU
        d_out = self.fc1.backward(d_out)  # Обратный проход через первый полносвязный слой
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        # Add regularization to the loss
        for param_name, param in params.items():
            if param_name.endswith('_W'):
                loss += 0.5 * self.reg * np.sum(param.value ** 2)  # Добавление регуляризации L2 к потерям
                params[param_name].grad += self.reg * param.value  # Добавление градиента регуляризации L2 к градиентам параметров
        

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int64) 
        out = self.fc1.forward(X)  # Прямой проход через первый полносвязный слой
        out = self.relu1.forward(out)  # Применение слоя активации ReLU
        scores = self.fc2.forward(out)  # Прямой проход через второй полносвязный слой
        
        # Predict
        pred = np.argmax(scores, axis=1)  # Предсказание меток классов
        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params
        result = {'fc1_W': self.fc1.params()['W'], 'fc1_B': self.fc1.params()['B'],
                  'fc2_W': self.fc2.params()['W'], 'fc2_B': self.fc2.params()['B']}
        

        return result
