import numpy as np
import pandas as pd

class DataPrep(object):
    def __init__(self, df, y):
        self.df = df.copy()
        self.y = y.copy()
        self.n_data = df.shape[0]
        self.arrs = []
        self.cols = []
        self.nans = []
    
    def search_nan(self):
        for i in self.df:
            for j in self.df[i]:
                if j != j:
                    self.nans.append(i)
                    break

    def train_test_split(self, test_size=0.15, val_size=0.15):
        if self.prepared_df is not None:
            n_train = int(self.n_data*(1 - test_size - val_size))
            n_test = int(self.n_data*(test_size))
            n_val = self.n_data - n_train - n_test
            arr_data = self.prepared_df.to_numpy()
            arr_y = self.y
            np.random.seed(31)
            p = np.random.permutation(len(self.y))
            arr_data = arr_data[p]
            arr_y = arr_y[p]
            if val_size == 0:
                return arr_data[:n_train], arr_y[:n_train], arr_data[n_train:], arr_y[n_train:]
            else:
                return arr_data[:n_train], arr_y[:n_train], arr_data[n_train:n_train+n_val], arr_y[n_train:n_train+n_val], arr_data[n_train+n_val:], arr_y[n_train+n_val:]
        else:
            print("Run prepare() first!!!")
        
    def fill_categorical_nan_and_one_hot_encode(self, col):
        arr = self.df[col].to_numpy()
        dct = dict()
        nan_ind = []

        for i in range(len(arr)):
            element = arr[i]
            if isinstance(element, str) and element != 'nan':
                try:
                    dct[element] += 1
                except:
                    dct[element] = 1
            else:
                nan_ind.append(i)

        nonnan_len = len(arr) - len(nan_ind)
        for i in dct:
            dct[i] = dct[i]/nonnan_len
        for i in nan_ind:
            # arr[i] = np.random.choice(list(dct.keys()), p=list(dct.values()))
            arr[i] = "nan"
        freq_arr = np.zeros(len(dct.keys()))
        uni_vals = np.unique(arr).tolist()
        uni_vals.remove("nan")
        encoded_arr = np.zeros((self.n_data, len(uni_vals)))
        encod_dict = dict()
        for i in range(len(uni_vals)):
            encod_dict[uni_vals[i]] = i
        for i in range(len(freq_arr)):
            freq_arr[i] = dct[uni_vals[i]]
        for i in range(self.n_data):
            if arr[i] == "nan":
                encoded_arr[i] = freq_arr.copy()
            else:
                encoded_arr[i][encod_dict[arr[i]]] = 1
        return encoded_arr, encod_dict

    def fill_numerical_nan(self, col):
        return self.df[col].fillna(self.df[col].mean())

    def one_hot_encode_not_nan(self, col):
        arr = self.df[col].to_numpy()
        dct = dict()

        for i in range(len(arr)):
            element = arr[i]
            try:
                dct[element] += 1
            except:
                dct[element] = 1

        uni_vals = np.unique(arr).tolist()
        encoded_arr = np.zeros((self.n_data, len(uni_vals)))
        encod_dict = dict()
        for i in range(len(uni_vals)):
            encod_dict[uni_vals[i]] = i
        for i in range(self.n_data):
            encoded_arr[i][encod_dict[arr[i]]] = 1
        return encoded_arr, encod_dict

    def normalize_numerical(self, arr):
        return (arr - arr.mean())/arr.std()

    def prepare(self):
        d_types = self.df.dtypes
        self.search_nan()
        for i in self.df:
            if i in self.nans:
                if d_types[i] == "object":
                    ret_tuple = self.fill_categorical_nan_and_one_hot_encode(i)
                    for k in ret_tuple[0].T:
                        self.arrs.append(k)
                    for j in ret_tuple[1]:
                        self.cols.append(i+'_'+j)
                else:
                    self.arrs.append(self.normalize_numerical(self.fill_numerical_nan(i)))
                    self.cols.append(i)
            else:
                if d_types[i] == "object":
                    ret_tuple = self.one_hot_encode_not_nan(i)
                    for k in ret_tuple[0].T:
                        self.arrs.append(k)
                    for j in ret_tuple[1]:
                        self.cols.append(i+'_'+j)
                else:
                    self.arrs.append(self.normalize_numerical(self.df[i]))
                    self.cols.append(i)
        self.prepared_df = pd.DataFrame(np.array(self.arrs).T, columns=self.cols)
        return pd.DataFrame(np.array(self.arrs).T, columns=self.cols), self.y

## Example Usage

# preparator = DataPrep(Dataframe, labels)
# PreparedDataframe, labels = preparator.prepare()
# X_train, y_train, X_val, y_val, X_test, y_test = preparator.train_test_split(test_size, val_size)