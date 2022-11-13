import unittest
# from iwanttobeanengineersexcanwait.FullyConnected import FullyConnected
from tramboline import PyFC as FullyConnected
from tramboline import PyTensor as t
from iwanttobeanengineersexcanwait import SGD
from iwanttobeanengineersexcanwait import L2Loss
import Helpers
import numpy as np

class TestFullyConnected1(unittest.TestCase):
    def setUp(self):
        self.batch_size = 9
        self.input_size = 4
        self.output_size = 3
        self.input_tensor = t(np.random.rand(self.batch_size, self.input_size)[:, :, np.newaxis].tolist())

        self.categories = 4
        self.label_tensor = t(np.zeros([self.batch_size, self.categories])[:, :, np.newaxis].tolist())
        for i in range(self.batch_size):
            self.label_tensor[0, i, np.random.randint(0, self.categories)] = 1  # one-hot encoded labels

    def test_trainable(self):
        layer = FullyConnected(self.input_size, self.output_size)
        self.assertTrue(layer.trainable)

    def test_forward_size(self):
        layer = FullyConnected(self.input_size, self.output_size)
        output_tensor = layer.forward(self.input_tensor)
        self.assertEqual(output_tensor.shape[1], self.output_size)
        self.assertEqual(output_tensor.shape[0], self.batch_size)

    def test_backward_size(self):
        layer = FullyConnected(self.input_size, self.output_size)
        output_tensor = layer.forward(self.input_tensor)
        # print(output_tensor.shape)
        error_tensor = layer.backward(output_tensor)
        self.assertEqual(error_tensor.shape[1], self.input_size)
        self.assertEqual(error_tensor.shape[0], self.batch_size)

    def test_update(self):
        layer = FullyConnected(self.input_size, self.output_size, 1, 'SGD', 0, 0)
        for _ in range(10):
            output_tensor = layer.forward(self.input_tensor)
            error_tensor = t(np.zeros([self.batch_size, self.output_size, 1]).tolist())
            # print(error_tensor.shape, output_tensor.shape)
            error_tensor = error_tensor - output_tensor
            # print(error_tensor.shape)
            layer.backward(error_tensor)
            new_output_tensor = layer.forward(self.input_tensor)
            self.assertLess(output_tensor.__pow__(2).sum(), new_output_tensor.__pow__(2).sum())

    def test_update_bias(self):
        input_tensor = t(np.zeros([self.batch_size, self.input_size, 1]).tolist())
        layer = FullyConnected(self.input_size, self.output_size, 1, 'SGD', 0, 0)
        # layer.optimizer = SGD(1)
        for _ in range(10):
            output_tensor = layer.forward(input_tensor)
            error_tensor = t(np.zeros([self.batch_size, self.output_size, 1]).tolist())
            error_tensor = error_tensor - output_tensor
            layer.backward(error_tensor)
            new_output_tensor = layer.forward(input_tensor)
            self.assertLess((output_tensor ** 2).sum(), (new_output_tensor ** 2).sum())

    def test_gradient(self):
        input_tensor = t(np.abs(np.random.random((self.batch_size, self.input_size, 1))).tolist())
        layers = list()
        layers.append(FullyConnected(self.input_size, self.categories))
        layers.append(L2Loss())
        difference = Helpers.gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(difference.sum(), 1e-5)

    def test_gradient_weights(self):
        input_tensor = t(np.abs(np.random.random((self.batch_size, self.input_size, 1))).tolist())
        layers = list()
        layers.append(FullyConnected(self.input_size, self.categories))
        layers.append(L2Loss())
        difference = Helpers.gradient_check_weights(layers, input_tensor, self.label_tensor, False)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_bias(self):
        input_tensor = t(np.zeros((1, 100000, 1)).tolist())
        layer = FullyConnected(100000, 1)
        result = layer.forward(input_tensor)
        self.assertGreater(result.sum(), 0)
