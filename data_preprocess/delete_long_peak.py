#!/usr/bin/python
import sys

file_name = sys.argv[1]
length = sys.argv[2]
fout_name = file_name.replace(".bed",".delete.bed")
fout = open(fout_name,"w")
for line in open(file_name):
    cordinate = line.strip().split("\t")
    if (long(cordinate[2]) - long(cordinate[1])) <= length:
        fout.write(line)
fout.close()
