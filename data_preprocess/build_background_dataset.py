#!/usr/bin/python
import numpy as np

background_sequence = np.load("background_million.bed.matrix.npy")
row,col = background_sequence.shape
background_label = np.zeros((row, 1),dtype=background_sequence.dtype)

background_label_seq = np.c_[background_sequence,background_label]

np.save("background_label_seq",background_label_seq)
