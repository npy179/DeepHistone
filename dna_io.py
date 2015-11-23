#!/usr/bin/python
import numpy as np
import glob

def main():
    fastafiles = glob.glob("*.fa")
    for i in range(len(fastafiles)):
        print "working on: "+fastafiles[i]
        fastafile = open(fastafiles[i])
        seqs_list = []
        seqs_label = []
        label_count = 0
        for line in fastafile:
            if line.startswith(">"):
                continue
            else:
                seq = line.strip()
                seq_vec = dna_one_vector(seq,seq_len=600)
                label_count += 1
                seqs_list.append(seq_vec)
        seqs_matrix = np.vstack(seqs_list)
        seqs_label = np.ones((label_count,1),dtype='int8')*(i+1)
        matrix_name = fastafiles[i].replace("fa","matrix")
        seqs_label_name = fastafiles[i].replace("fa","label")
        np.save(matrix_name,seqs_matrix)
        np.save(seqs_label_name,seqs_label)

def dna_one_vector(seq, seq_len=None):
    seq = seq.upper()
    if seq_len == None:
        seq_len = len(seq)
        seq_start = 0
    else:
        if seq_len <= len(seq):
            #trim the sequence
            seq_trim = (len(seq)-seq_len)/2
            seq = seq[seq_trim:seq_trim+seq_len]
            seq_start = 0
        else:
            #if sequence length is bigger than the set length, set the begin and end as 0.25
            seq_start = (seq_len-len(seq))/2

    seq = seq.replace('A','0')
    seq = seq.replace('C','1')
    seq = seq.replace('G','2')
    seq = seq.replace('T','3')

    # map A,C,G,T to [1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]
    # allocate the seq_code with
    seq_code = np.zeros((4,seq_len),dtype='float16')
    for i in range(seq_len):
        if i < seq_start:
            seq_code[:,i] = 0.25
        else:
            try:
                seq_code[int(seq[i-seq_start]),i] = 1
            except:
                seq_code[:,i] = 0.25

    #flatten and make vector as 1 * len(seq)
    seq_vec = seq_code.flatten()[None,:]
    return seq_vec


if __name__=="__main__":
    main()
