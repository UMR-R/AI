import numpy as np
import h5py
import matplotlib.pyplot as plt

def load_dataset():
    train_dataset = h5py.File("datasets/train_face.h5", "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig