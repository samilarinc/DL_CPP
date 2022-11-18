import unittest
try:
    LSTM_TEST = True
    from Layers import *
except BaseException as e:
    if str(e)[-6:] == "'LSTM'":
        LSTM_TEST = False
    else:
        raise e
from pyflow import *
import numpy as np
from scipy import stats
from scipy.ndimage.filters import gaussian_filter
import NeuralNetwork
import matplotlib.pyplot as plt
import os
import Helpers

ID = 3  # identifier for dispatcher

class TestFullyConnected(unittest.TestCase):
    def setUp(self):
        self.batch_size = 9
        self.input_size = 4
        self.output_size = 3
        self.input_tensor = np.random.rand(self.batch_size, self.input_size)

        self.categories = 4
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    class TestInitializer:
        def __init__(self):
            self.fan_in = None
            self.fan_out = None

        def initialize(self, shape, fan_in, fan_out):
            self.fan_in = fan_in
            self.fan_out = fan_out
            weights = np.zeros(shape)
            weights[0] = 1
            weights[1] = 2
            return weights

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
        error_tensor = layer.backward(output_tensor)
        self.assertEqual(error_tensor.shape[1], self.input_size)
        self.assertEqual(error_tensor.shape[0], self.batch_size)

    def test_update(self):
        layer = FullyConnected(self.input_size, self.output_size, 1, 'SGD', 0, 0)
        for _ in range(10):
            output_tensor = layer.forward(self.input_tensor)
            error_tensor = np.zeros([ self.batch_size, self.output_size])
            error_tensor -= output_tensor
            layer.backward(error_tensor)
            new_output_tensor = layer.forward(self.input_tensor)
            self.assertLess(np.sum(np.power(output_tensor, 2)), np.sum(np.power(new_output_tensor, 2)))

    def test_update_bias(self):
        input_tensor = np.zeros([self.batch_size, self.input_size])
        layer = FullyConnected(self.input_size, self.output_size, 1, 'SGD', 0, 0)
        for _ in range(10):
            output_tensor = layer.forward(input_tensor)
            error_tensor = np.zeros([self.batch_size, self.output_size])
            error_tensor -= output_tensor
            layer.backward(error_tensor)
            new_output_tensor = layer.forward(input_tensor)
            self.assertLess(np.sum(np.power(output_tensor, 2)), np.sum(np.power(new_output_tensor, 2)))

    def test_gradient(self):
        input_tensor = np.abs(np.random.random((self.batch_size, self.input_size)))
        layers = list()
        layers.append(FullyConnected(self.input_size, self.categories))
        layers.append(L2Loss())
        difference = Helpers.gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_gradient_weights(self):
        input_tensor = np.abs(np.random.random((self.batch_size, self.input_size)))
        layers = list()
        layers.append(FullyConnected(self.input_size, self.categories))
        layers.append(L2Loss())
        difference = Helpers.gradient_check_weights(layers, input_tensor, self.label_tensor, False)
        self.assertLessEqual(np.sum(difference), 1e-5)

    def test_bias(self):
        input_tensor = np.zeros((1, 100000))
        layer = FullyConnected(100000, 1)
        result = layer.forward(input_tensor)
        self.assertGreater(np.sum(result), 0)

    def test_initialization(self):
        input_size = 4
        categories = 10
        layer = FullyConnected(input_size, categories)
        init = TestFullyConnected.TestInitializer()
        layer.initialize(init, Initializers.Constant(0.5))
        self.assertEqual(init.fan_in, input_size)
        self.assertEqual(init.fan_out, categories)
        if layer.weights.shape[0]>layer.weights.shape[1]:
            self.assertLessEqual(np.sum(layer.weights) - 17, 1e-5)
        else:
            self.assertLessEqual(np.sum(layer.weights) - 35, 1e-5)

class TestReLU(unittest.TestCase):
    def setUp(self):
        self.input_size = 5
        self.batch_size = 10
        self.half_batch_size = int(self.batch_size / 2)
        self.input_tensor = np.ones([self.batch_size, self.input_size])
        self.input_tensor[0:self.half_batch_size,:] -= 2

        self.label_tensor = np.zeros([self.batch_size, self.input_size])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.input_size)] = 1

    def test_trainable(self):
        layer = ReLU.ReLU()
        self.assertFalse(layer.trainable)

    def test_forward(self):
        expected_tensor = np.zeros([self.batch_size, self.input_size])
        expected_tensor[self.half_batch_size:self.batch_size, :] = 1

        layer = ReLU.ReLU()
        output_tensor = layer.forward(self.input_tensor)
        self.assertEqual(np.sum(np.power(output_tensor-expected_tensor, 2)), 0)

    def test_backward(self):
        expected_tensor = np.zeros([self.batch_size, self.input_size])
        expected_tensor[self.half_batch_size:self.batch_size, :] = 2

        layer = ReLU.ReLU()
        layer.forward(self.input_tensor)
        output_tensor = layer.backward(self.input_tensor*2)
        self.assertEqual(np.sum(np.power(output_tensor - expected_tensor, 2)), 0)

    def test_gradient(self):
        input_tensor = np.abs(np.random.random((self.batch_size, self.input_size)))
        input_tensor *= 2.
        input_tensor -= 1.
        layers = list()
        layers.append(ReLU.ReLU())
        layers.append(L2Loss())
        difference = Helpers.gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5)

class TestCrossEntropyLoss(unittest.TestCase):

    def setUp(self):
        self.batch_size = 9
        self.categories = 4
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    def test_gradient(self):
        input_tensor = np.abs(np.random.random(self.label_tensor.shape))
        layers = list()
        layers.append(Loss.CrossEntropyLoss())
        difference = Helpers.gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-4)

    def test_zero_loss(self):
        layer = Loss.CrossEntropyLoss()
        loss = layer.forward(self.label_tensor, self.label_tensor)
        self.assertAlmostEqual(loss, 0)

    def test_high_loss(self):
        label_tensor = np.zeros((self.batch_size, self.categories))
        label_tensor[:, 2] = 1
        input_tensor = np.zeros_like(label_tensor)
        input_tensor[:, 1] = 1
        layer = Loss.CrossEntropyLoss()
        loss = layer.forward(input_tensor, label_tensor)
        self.assertAlmostEqual(loss, 324.3928805, places = 4)

class TestNeuralNetwork3(unittest.TestCase):
    plot = False
    directory = 'plots/'
    log = 'log.txt'
    iterations = 100

    def test_append_layer(self):
        # this test checks if your network actually appends layers, whether it copies the optimizer to these layers, and
        # whether it handles the initialization of the layer's weights
        net = NeuralNetwork.NeuralNetwork('SGD', 1)
        fcl_1 = FullyConnected(1, 1, 1, 'SGD', 0, 0)
        net.append_layer(fcl_1)
        fcl_2 = FullyConnected(1, 1, 1, 'SGD', 0, 0)
        net.append_layer(fcl_2)

        self.assertEqual(len(net.layers), 2)
        self.assertFalse(net.layers[0].optimizer is net.layers[1].optimizer)
        self.assertTrue(np.all(net.layers[0].weights == 0.123))

    def test_data_access(self):
        net = NeuralNetwork.NeuralNetwork('SGD', 1e-4)
        categories = 3
        input_size = 4
        net.data_layer = Helpers.IrisData(50)
        net.loss_layer = CrossEntropyLoss()
        fcl_1 = FullyConnected(input_size, categories, 1e-4, 'SGD', 0, 0)
        net.append_layer(fcl_1)
        net.append_layer(ReLU())
        fcl_2 = FullyConnected(categories, categories, 1e-4, 'SGD', 0, 0)
        net.append_layer(fcl_2)
        net.append_layer(SoftMax())

        out = net.forward()
        out2 = net.forward()

        self.assertNotEqual(out, out2)

    def test_iris_data(self):
        net = NeuralNetwork.NeuralNetwork(SGD(1e-3),
                                          Initializers.UniformRandom(),
                                          Initializers.Constant(0.1))
        categories = 3
        input_size = 4
        net.data_layer = Helpers.IrisData(100)
        net.loss_layer = Loss.CrossEntropyLoss()
        fcl_1 = FullyConnected(input_size, categories)
        net.append_layer(fcl_1)
        net.append_layer(ReLU.ReLU())
        fcl_2 = FullyConnected(categories, categories)
        net.append_layer(fcl_2)
        net.append_layer(SoftMax.SoftMax())

        net.train(4000)
        if TestNeuralNetwork3.plot:
            fig = plt.figure('Loss function for a Neural Net on the Iris dataset using SGD')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestNeuralNetwork3.pdf"), transparent=True, bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = Helpers.calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the Iris dataset, we achieve an accuracy of: ' + str(accuracy * 100) + '%', file=f)
        self.assertGreater(accuracy, 0.9)

    def test_regularization_loss(self):
        '''
        This test checks if the regularization loss is calculated for the fc and rnn layer and tracked in the
        NeuralNetwork class
        '''
        import random
        fcl = FullyConnected(4, 3)
        rnn = RNN.RNN(4, 4, 3)

        for layer in [fcl, rnn]:
            loss = []
            for reg in [False, True]:
                opt = SGD(1e-3)
                if reg:
                    opt.add_regularizer(Constraints.L1_Regularizer(8e-2))
                net = NeuralNetwork.NeuralNetwork(opt,Initializers.Constant(0.5),
                                                      Initializers.Constant(0.1))

                net.data_layer = Helpers.IrisData(100, random = False)
                net.loss_layer = Loss.CrossEntropyLoss()
                net.append_layer(layer)
                net.append_layer(SoftMax.SoftMax())
                net.train(1)
                loss.append(np.sum(net.loss))

            self.assertNotEqual(loss[0], loss[1], "Regularization Loss is not calculated and added to the overall loss "
                                                  "for " + layer.__class__.__name__)

    def test_iris_data_with_momentum(self):
        net = NeuralNetwork.NeuralNetwork(SGDWithMomentum(1e-3, 0.8),
                                          Initializers.UniformRandom(),
                                          Initializers.Constant(0.1))
        categories = 3
        input_size = 4
        net.data_layer = Helpers.IrisData(100)
        net.loss_layer = Loss.CrossEntropyLoss()
        fcl_1 = FullyConnected(input_size, categories)
        net.append_layer(fcl_1)
        net.append_layer(ReLU.ReLU())
        fcl_2 = FullyConnected(categories, categories)
        net.append_layer(fcl_2)
        net.append_layer(SoftMax.SoftMax())

        net.train(2000)
        if TestNeuralNetwork3.plot:
            fig = plt.figure('Loss function for a Neural Net on the Iris dataset using Momentum')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestNeuralNetwork3_Momentum.pdf"), transparent=True, bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = Helpers.calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the Iris dataset, we achieve an accuracy of: ' + str(accuracy * 100) + '%', file=f)
        self.assertGreater(accuracy, 0.9)

    def test_iris_data_with_adam(self):
        net = NeuralNetwork.NeuralNetwork(Optimizers.Adam(1e-3, 0.9, 0.999),
                                          Initializers.UniformRandom(),
                                          Initializers.Constant(0.1))
        categories = 3
        input_size = 4
        net.data_layer = Helpers.IrisData(100)
        net.loss_layer = Loss.CrossEntropyLoss()
        fcl_1 = FullyConnected(input_size, categories)
        net.append_layer(fcl_1)
        net.append_layer(ReLU.ReLU())
        fcl_2 = FullyConnected(categories, categories)
        net.append_layer(fcl_2)
        net.append_layer(SoftMax.SoftMax())

        net.train(3000)
        if TestNeuralNetwork3.plot:
            fig = plt.figure('Loss function for a Neural Net on the Iris dataset using ADAM')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestNeuralNetwork3_ADAM.pdf"), transparent=True, bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = Helpers.calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the Iris dataset, we achieve an accuracy of: ' + str(accuracy * 100) + '%', file=f)
        self.assertGreater(accuracy, 0.9)

    def test_iris_data_with_batchnorm(self):
        net = NeuralNetwork.NeuralNetwork(Optimizers.Adam(1e-2, 0.9, 0.999),
                                          Initializers.UniformRandom(),
                                          Initializers.Constant(0.1))
        categories = 3
        input_size = 4
        net.data_layer = Helpers.IrisData(50)
        net.loss_layer = Loss.CrossEntropyLoss()
        net.append_layer(BatchNormalization.BatchNormalization(input_size))
        fcl_1 = FullyConnected(input_size, categories)
        net.append_layer(fcl_1)
        net.append_layer(ReLU.ReLU())
        fcl_2 = FullyConnected(categories, categories)
        net.append_layer(fcl_2)
        net.append_layer(SoftMax.SoftMax())

        net.train(2000)
        if TestNeuralNetwork3.plot:
            fig = plt.figure('Loss function for a Neural Net on the Iris dataset using Batchnorm')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestNeuralNetwork3_Batchnorm.pdf"), transparent=True, bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        results_next_run = net.test(data)

        accuracy = Helpers.calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the Iris dataset using Batchnorm, we achieve an accuracy of: ' + str(accuracy * 100.) + '%', file=f)
        self.assertGreater(accuracy, 0.8)
        self.assertEqual(np.mean(np.square(results - results_next_run)), 0)

    def test_iris_data_with_dropout(self):
        net = NeuralNetwork.NeuralNetwork(Optimizers.Adam(1e-2, 0.9, 0.999),
                                          Initializers.UniformRandom(),
                                          Initializers.Constant(0.1))
        categories = 3
        input_size = 4
        net.data_layer = Helpers.IrisData(50)
        net.loss_layer = Loss.CrossEntropyLoss()
        fcl_1 = FullyConnected(input_size, categories)
        net.append_layer(fcl_1)
        net.append_layer(ReLU.ReLU())
        fcl_2 = FullyConnected(categories, categories)
        net.append_layer(fcl_2)
        net.append_layer(Dropout.Dropout(0.3))
        net.append_layer(SoftMax.SoftMax())

        net.train(2000)
        if TestNeuralNetwork3.plot:
            fig = plt.figure('Loss function for a Neural Net on the Iris dataset using Dropout')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestNeuralNetwork3_Dropout.pdf"), transparent=True, bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = Helpers.calculate_accuracy(results, labels)

        results_next_run = net.test(data)

        with open(self.log, 'a') as f:
            print('On the Iris dataset using Dropout, we achieve an accuracy of: ' + str(accuracy * 100.) + '%', file=f)
        self.assertEqual(np.mean(np.square(results - results_next_run)), 0)

    def test_layer_phases(self):
        net = NeuralNetwork.NeuralNetwork(Optimizers.Adam(1e-2, 0.9, 0.999),
                                          Initializers.UniformRandom(),
                                          Initializers.Constant(0.1))
        categories = 3
        input_size = 4
        net.data_layer = Helpers.IrisData(50)
        net.loss_layer = Loss.CrossEntropyLoss()
        net.append_layer(BatchNormalization.BatchNormalization(input_size))
        fcl_1 = FullyConnected(input_size, categories)
        net.append_layer(fcl_1)
        net.append_layer(ReLU.ReLU())
        fcl_2 = FullyConnected(categories, categories)
        net.append_layer(fcl_2)
        net.append_layer(Dropout.Dropout(0.3))
        net.append_layer(SoftMax.SoftMax())

        net.train(100)

        data, labels = net.data_layer.get_test_set()
        results = net.test(data)

        bn_phase = net.layers[0].testing_phase
        drop_phase = net.layers[4].testing_phase

        self.assertTrue(bn_phase)
        self.assertTrue(drop_phase)

    def test_digit_data(self):
        adam = Optimizers.Adam(5e-3, 0.98, 0.999)
        self._perform_test(adam, TestNeuralNetwork3.iterations, 'ADAM', False, False)

    def test_digit_data_L2_Regularizer(self):
        sgd_with_l2 = Optimizers.Adam(5e-3, 0.98, 0.999)
        sgd_with_l2.add_regularizer(Constraints.L2_Regularizer(8e-2))
        self._perform_test(sgd_with_l2, TestNeuralNetwork3.iterations, 'L2_regularizer', False, False)

    def test_digit_data_L1_Regularizer(self):
        sgd_with_l1 = Optimizers.Adam(5e-3, 0.98, 0.999)
        sgd_with_l1.add_regularizer(Constraints.L1_Regularizer(8e-2))
        self._perform_test(sgd_with_l1, TestNeuralNetwork3.iterations, 'L1_regularizer', False, False)

    def test_digit_data_dropout(self):
        sgd_with_l2 = Optimizers.Adam(5e-3, 0.98, 0.999)
        sgd_with_l2.add_regularizer(Constraints.L2_Regularizer(4e-4))
        self._perform_test(sgd_with_l2, TestNeuralNetwork3.iterations, 'Dropout', True, False)

    def test_digit_batch_norm(self):
        adam = Optimizers.Adam(1e-2, 0.98, 0.999)
        self._perform_test(adam, TestNeuralNetwork3.iterations, 'Batch_norm', False, True)

    def test_all(self):
        sgd_with_l2 = Optimizers.Adam(1e-2, 0.98, 0.999)
        sgd_with_l2.add_regularizer(Constraints.L2_Regularizer(8e-2))
        self._perform_test(sgd_with_l2, TestNeuralNetwork3.iterations, 'Batch_norm and L2', False, True)

    def _perform_test(self, optimizer, iterations, description, dropout, batch_norm):
        net = NeuralNetwork.NeuralNetwork(optimizer,
                                          Initializers.He(),
                                          Initializers.Constant(0.1))
        input_image_shape = (1, 8, 8)
        conv_stride_shape = (1, 1)
        convolution_shape = (1, 3, 3)
        categories = 10
        batch_size = 150
        num_kernels = 4

        net.data_layer = Helpers.DigitData(batch_size)
        net.loss_layer = Loss.CrossEntropyLoss()

        if batch_norm:
            net.append_layer(BatchNormalization.BatchNormalization(1))

        cl_1 = Conv.Conv(conv_stride_shape, convolution_shape, num_kernels)
        net.append_layer(cl_1)
        cl_1_output_shape = (num_kernels, *input_image_shape[1:])

        if batch_norm:
            net.append_layer(BatchNormalization.BatchNormalization(num_kernels))

        net.append_layer(ReLU.ReLU())

        fcl_1_input_size = np.prod(cl_1_output_shape)

        net.append_layer(Flatten.Flatten())

        fcl_1 = FullyConnected(fcl_1_input_size, int(fcl_1_input_size/2.))
        net.append_layer(fcl_1)

        if batch_norm:
            net.append_layer(BatchNormalization.BatchNormalization(fcl_1_input_size//2))

        if dropout:
            net.append_layer(Dropout.Dropout(0.3))

        net.append_layer(ReLU.ReLU())

        fcl_2 = FullyConnected(int(fcl_1_input_size / 2), int(fcl_1_input_size / 3))
        net.append_layer(fcl_2)

        net.append_layer(ReLU.ReLU())

        fcl_3 = FullyConnected(int(fcl_1_input_size / 3), categories)
        net.append_layer(fcl_3)

        net.append_layer(SoftMax.SoftMax())

        net.train(iterations)
        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = Helpers.calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the UCI ML hand-written digits dataset using {} we achieve an accuracy of: {}%'.format(description, accuracy * 100.), file=f)
        print('\nOn the UCI ML hand-written digits dataset using {} we achieve an accuracy of: {}%'.format(description, accuracy * 100.))
        self.assertGreater(accuracy, 0.3)

if __name__ == "__main__":

    import sys
    if sys.argv[-1] == "Bonus":
        loader = unittest.TestLoader()
        bonus_points = {}
        tests = [TestNeuralNetwork3, TestCrossEntropyLoss, TestReLU, TestFullyConnected]
        percentages = [5, 5, 5, 5]
        total_points = 0
        for t, p in zip(tests, percentages):
            if unittest.TextTestRunner().run(loader.loadTestsFromTestCase(t)).wasSuccessful():
                bonus_points.update({t.__name__: ["OK", p]})
                total_points += p
            else:
                bonus_points.update({t.__name__: ["FAIL", p]})
    else:
        unittest.main()
