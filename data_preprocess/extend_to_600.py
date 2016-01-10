#!/usr/bin/python
import glob
import os
import sys

dir = sys.argv[1]
os.chdir(dir)

files = glob.glob("*.bed")
for fil in files:
    fin = open(fil)
    founame = fil.replace(".bed","_600.bed")
    fout = open(founame, "w")
    for line in fin:
        content = line.split("\t")
        mid = int((float(content[1])+float(content[2]))/2)
        start = mid-300
        if start < 0:
            start = 0
        end = mid+300
        content[1] = str(start)
        content[2] = str(end)
        newline = "\t".join(content)
        fout.write(newline)
