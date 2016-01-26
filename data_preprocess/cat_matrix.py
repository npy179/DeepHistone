#!/usr/bin/python
import numpy as np
import sys


seq_label_back = np.load("background_label_seq.npy")
seq_label_fore_name = sys.argv[1]

seq_label_fore = np.load(seq_label_fore_name)

row, col = seq_label_fore.shape

seq_label_fore_back = np.r_[seq_label_back, seq_label_fore]

np.random.shuffle(seq_label_fore_back)

seq_fore_back = seq_label_fore_back[:,:-1]
label_fore_back_float16 = seq_label_fore_back[:,-1]
label_fore_back_int64 = label_fore_back_float16.astype("int64")


whole_sequence_name = seq_label_fore_name.split("_")[0]+"_sequence_fore_back"
whole_label_name = seq_label_fore_name.split("_")[0]+"_label_fore_back"
np.save(whole_sequence_name, seq_fore_back)
np.save(whole_label_name, label_fore_back_int64)

#print whole_sequence_name
#print whole_label_name
