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
    
    def train(self, epochs, cross_val = False, valset = None, verbose = True):
        self.vallosses = list()
        if cross_val and not verbose:
            raise ValueError("Cross validation must be verbose")
        for epoch in range(epochs):
            inner_loss = 0
            iterations = 0
            while not self.data.finished:
                loss = self.forward()
                inner_loss += loss
                self.backward()
                iterations += 1
            inner_loss /= iterations
            self.loss.append(inner_loss)
            self.data.finished = False
            if verbose:
                if cross_val:
                    valloss = (self.test(valset[0]) - valset[1]).power(2).sum() / valset[1].shape[0]
                print("Epoch: %4d\n\tTrain Loss: %6.2f"%(epoch+1, inner_loss) + (("\tVal Loss: %6.2f"%valloss) if cross_val else ""))
            self.vallosses.append(valloss)
        if cross_val:
            return self.loss, self.vallosses
        return self.loss
            
    def test(self, data):
        self.testing_phase = True
        X = t(data[:, :, None].tolist())
        for layer in self.layers:
            X = layer.forward(X)
        return X
    