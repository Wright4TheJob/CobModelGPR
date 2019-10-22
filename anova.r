# two-way ANOVA for cob compressive data
library(car)

data <- read.csv("SizeExperimentBending.txt",sep="\t")
# print(data)

# print(Anova(lm(Strength ~ Clay*Straw, data), type="3"))
print(anova(lm(Peak ~ Batch*Size, data)))
