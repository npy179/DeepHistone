import os
import sys
import glob

bedfiles = glob.glob("*.bed")
for i in range(0,len(bedfiles)):
    for j in range(0, len(bedfiles)):
        new_overlape_name = bedfiles[i].replace("E017-","").replace(".bed","")+"_"+bedfiles[j].replace("E017-","").replace(".bed","")
        exc = "bedtools intersect "+"-a "+bedfiles[i]+" -b "+bedfiles[j]+" > "+ new_overlape_name
        os.system(exc)
