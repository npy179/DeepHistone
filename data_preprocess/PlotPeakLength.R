setwd("/Users/pni/DeepHistone/E017raw/E017Raw")
cordinate <- read.table("E017.txt",sep="\t",quote = "\"")
length <- cordinate[2]-cordinate[1]
length_num <- as.numeric(as.matrix(length))
typeof(length_num)
hist(length_num,breaks = 10000)
max(length_num)
min(length_num)
mean(length_num)
length_small <- length_num[length_num<2000]
hist(length_small,breaks = 100)
