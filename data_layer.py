import pandas as pd
import numpy as np

class Dataset(object):
    # def __init__(self, data_dir, batch_size, label_name): #Habitability_score
    #     self.data_dir = data_dir
    #     self.batch_size = batch_size
    #     self.label_name = label_name
    #     self.load_data()
    #     self.data_len = self.data.shape[0]
    #     self.finished = False
    #     self.last_loc = 0
    
    def __init__(self, X, y, batch_size, from_numpy=True):
        self.data = X
        self.labels = y
        self.finished = False
        self.data_len = self.data.shape[0]
        self.batch_size = batch_size
        self.last_loc = 0
    
    # def load_data(self):
    #     self.data = pd.read_csv(self.data_dir)
    #     self.labels = self.data[self.label_name].to_numpy()
    #     self.data = self.data.drop(self.label_name, axis=1).to_numpy()
    
    def next_batch(self):
        if self.finished:
            return None, None
        if self.batch_size > self.data.shape[0]:
            self.batch_size = self.data.shape[0]
            print("Batch size is too large. Batch size is set to {} which is the data length.".format(self.batch_size))
            return None, None
        if self.last_loc + self.batch_size >= self.data.shape[0]:
            indices = np.arange(self.last_loc, self.data.shape[0], 1).tolist()
            indices.extend(np.arange(0, self.batch_size - (self.data.shape[0] - self.last_loc), 1).tolist())
            indices = np.array(indices)
            self.last_loc = 0
            self.finished = True
        else:
            indices = np.arange(self.last_loc, self.last_loc + self.batch_size, 1)
            self.last_loc += self.batch_size
        return self.data[indices], self.labels[indices]
