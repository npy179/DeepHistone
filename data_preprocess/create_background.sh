bedtools random -l 600 -n 2000000 -g hg19.chrom.sizes.txt > background.bed
bedtools sort -i background.bed > background.sort.bed
bedtools intersect -v -wa -wb -a background.sort.bed -b E017-H3K27me3.bed.sort E017-H3K36me3.bed.sort E017-H3K4me1.bed.sort E017-H3K4me3.bed.sort E017-H3K9me3.bed.sort > unoverlap.bed
