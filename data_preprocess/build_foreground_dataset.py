#!/usr/bin/python
import numpy as np
import sys

matrix_name = sys.argv[1]
out_matrix_name = matrix_name.split(".")[0]+"_sequence_label"


foreground_sequence = np.load(matrix_name)
row,col = foreground_sequence.shape
foreground_label = np.ones((row, 1),dtype=foreground_sequence.dtype)

foreground_label_seq = np.c_[foreground_sequence,foreground_label]

np.save(out_matrix_name, foreground_label_seq)

