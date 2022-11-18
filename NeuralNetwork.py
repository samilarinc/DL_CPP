from pyflow import Tensor as t

class NeuralNetwork(object):
    def __init__(self, optimizer):
        self.optimizer_str = optimizer
        self._testing_phase = False
        self.loss = list()
        self.layers = list()
        self.data = None
        self.loss_layer = None
    
    def append_layer(self, layer):
        self.layers.append(layer)
    
    def forward(self):
        X, y = self.data.next_batch()
        y = y[:, None, None]
        X = X[:, :, None]
        X, y = t(X.tolist()), t(y.tolist())
        self.labels = y
        for layer in self.layers:
            X = layer.forward(X)
        loss = self.loss_layer.forward(X, y)
        return loss

    def backward(self):
        y = self.labels
        X = self.loss_layer.backward(y)
        for layer in reversed(self.layers):
            X = layer.backward(X)
    
    def train(self, epochs):
        for epoch in range(epochs):
            inner_loss = 0
            while not self.data.finished:
                loss = self.forward()
                inner_loss += loss
                self.backward()
            print("Epoch: %2d, Loss: %6.2f appended!"%(epoch, inner_loss))
            self.loss.append(inner_loss)
            self.data.finished = False
        return self.loss
            
    def test(self, data):
        self.testing_phase = True
        X = data
        for layer in self.layers:
            X = layer.forward(X)
        return X
    