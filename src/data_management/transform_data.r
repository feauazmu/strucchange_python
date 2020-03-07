source("project_paths.r")

for(i in c("Grossarl", "GermanM1"))
  load(paste(PATH_IN_DATA, paste(i, ".rda", sep=""), sep="/"))

write.csv(Grossarl, file=paste(PATH_OUT_DATA, "Grossarl.csv", sep="/"))
write.csv(GermanM1, file=paste(PATH_OUT_DATA, "GermanM1.csv", sep="/"))
