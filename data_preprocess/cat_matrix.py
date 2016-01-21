#!/usr/bin/python
import numpy as np

seq_label_back = np.load("seq_label_background.npy")

seq_fore = np.load("H3K27me3.sort.merge.delete_600.bed.matrix.npy")
row, col = seq_fore.shape
label_fore = np.ones((row,1),dtype=seq_fore.dtype)
seq_label_fore = np.c_[seq_fore, label_fore]

seq_label_fore_back = np.r_[seq_label_back, seq_label_fore]

np.random.shuffle(seq_label_fore_back)

np.save("seq_label_fore_back",seq_label_fore_back)

seq_fore_back = seq_label_fore_back[:,:-1]
label_fore_back_float16 = seq_label_fore_back[:,-1]
label_fore_back_int64 = label_fore_back_float16.astype("int64")

np.save("seq_fore_back",seq_fore_back)
np.save("label_fore_back",label_fore_back_int64)
