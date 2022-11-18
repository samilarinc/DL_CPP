from NeuralNetwork import NeuralNetwork as NN
from data_layer import Dataset
from pyflow import Tensor, FullyConnected, ReLU, L2Loss
import pickle

# Load data
data_dir = "temp.csv"
batch_size = 2
label_name = "label"
data = Dataset(data_dir, batch_size, label_name)

# Create a neural network
nn = NN("SGD")
nn.data = data

# Create layers
input_size = 4
hidden_size = 2
output_size = 1
lr = 1e-4
layer1 = FullyConnected(input_size, 1, lr, 'SGD', 0, 0)
with open("weights.pkl", "rb") as f:
    w = pickle.load(f)
layer1.weights = Tensor(w)
activation = ReLU()
layer2 = FullyConnected(hidden_size, output_size, lr, 'SGD', 0, 0)

nn.append_layer(layer1)
# nn.append_layer(activation)
# nn.append_layer(layer2)

# Create loss layer
loss_layer = L2Loss()
nn.loss_layer = loss_layer

# Train
epochs = 10000
loss = nn.train(epochs)

# Save the model
w = layer1.weights.tolist()
with open('weights.pkl', 'wb') as f:
    pickle.dump(w, f)