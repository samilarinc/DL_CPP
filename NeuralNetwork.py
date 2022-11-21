from pyflow import *
import pickle
t = Tensor

class NeuralNetwork(object):
    def __init__(self, dataset, loss_layer=None, path=None):
        self._testing_phase = False
        self.loss = list()
        self.layers = list()
        self.dataset = dataset
        if loss_layer is not None:
            self.loss_layer = loss_layer
        elif path is not None:
            self.load_model(path)
        else:
            raise ValueError("Must specify either loss_layer or path")

    def append_layer(self, layer):
        self.layers.append(layer)
    
    def __forward(self):
        X, y = self.dataset.next_batch()
        y = y[:, None, None]
        X = X[:, :, None]
        X, y = t(X.tolist()), t(y.tolist())
        self.labels = y
        for layer in self.layers:
            X = layer.forward(X)
        loss = self.loss_layer.forward(X, y)
        return loss

    def __backward(self):
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
            while not self.dataset.finished:
                loss = self.__forward()
                inner_loss += loss
                self.__backward()
                iterations += 1
            inner_loss /= iterations
            self.loss.append(inner_loss)
            self.dataset.finished = False
            if verbose:
                if cross_val:
                    loss_type = self.loss_layer.__class__.__name__
                    if loss_type == 'L1Loss':
                        valloss = (self.test(valset[0]) - valset[1]).abs().sum() / valset[1].shape[0]
                    elif loss_type == 'L2Loss':
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

    def save_model(self, path):
        import os
        if os.path.exists(path):
            print("Path already exists, specify a new path")
            return
        os.system("mkdir " + path)
        for layer_num in range(len(self.layers)):
            self.layers[layer_num].save(path, layer_num)
        str_layers = [str(layer.__class__.__name__) for layer in self.layers]
        with open(path + '/' + 'model.info', 'wb') as f:
            pickle.dump(str_layers, f)
            pickle.dump(self.loss_layer.__class__.__name__, f)
    
    def load_model(self, path):
        with open(path + '/' + 'model.info', 'rb') as f:
            str_layers = pickle.load(f)
            self.loss_layer = pickle.load(f)
        self.layers = list()
        for str_layer in str_layers:
            layer = eval(str_layer)()
            layer.load(path, len(self.layers))
            self.layers.append(layer)
        self.loss_layer = eval(self.loss_layer)()

    def predict(self, X):
        return self.test(X)

    def __call__(self, X):
        return self.test(X)
        