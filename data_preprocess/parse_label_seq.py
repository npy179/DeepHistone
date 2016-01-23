#!/usr/bin/python
import numpy as np

seq_label = np.load("seq_label_fore_back.npy")
seq = seq_label[:,:-1]
label = seq_label[:,-1]
label = label.astype("int64")

np.save("sequence",seq)
np.save("label",label)
