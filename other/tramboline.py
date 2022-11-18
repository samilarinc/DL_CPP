from iwanttobeanengineersexcanwait import Tensor as t
from iwanttobeanengineersexcanwait import FullyConnected as fc
import copy

class PyTensor(t):
    def __init__(self, *args):
        super().__init__(*args)

    def __getitem__(self, indexes):
        if len(indexes) == 1:
            return self.getitem(indexes[0])
        elif len(indexes) == 2:
            return self.getitem(indexes[0], indexes[1])
        elif len(indexes) == 3:
            return self.getitem(indexes[0], indexes[1], indexes[2])
    
    def __setitem__(self, key, value):
        if len(key) == 1:
            self.setitem(key[0], value)
        elif len(key) == 2:
            self.setitem(key[0], key[1], value)
        elif len(key) == 3:
            self.setitem(key[0], key[1], key[2], value)
    
    def py_copy(self):
        return PyTensor(self.copy())

class PyFC(fc):
    def forward(self, input_tensor):
        return PyTensor(super().forward(input_tensor))
    def backward(self, error_tensor):
        return PyTensor(super().backward(error_tensor))
    
    @property
    def _weights(self):
        return PyTensor(self.weights)
    @_weights.setter
    def _weights(self, value):
        weights = value
    @property
    def _bias(self):
        return PyTensor(self.bias)
    @_bias.setter
    def _bias(self, value):
        bias = value