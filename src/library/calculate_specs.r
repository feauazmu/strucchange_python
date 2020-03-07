'
Creates the specs for testing.
The databases where copied from the examples in strucchange package.
'
source("project_paths.r")

library('strucchange')
library('jsonlite')


load(paste(PATH_IN_DATA, "Grossarl.rda", sep="/"))
formula = marriages ~ 1
bp <- breakpoints(formula, data = Grossarl, h = 0.1)
rid <- recresid(formula, data = Grossarl)
exportJson <- toJSON(summary(bp), force=TRUE, pretty = TRUE)
write(exportJson, paste(PATH_OUT_MODEL_SPECS, "Grossarl_bp.json", sep="/"))
exportJson <- toJSON(rid, force=TRUE, pretty = TRUE)
write(exportJson, paste(PATH_OUT_MODEL_SPECS, "Grossarl_rid.json", sep="/"))

load(paste(PATH_IN_DATA, "GermanM1.rda", sep="/"))
formula = dm ~ dy2 + dR + dR1 + dp + m1 + y1 + R1
bp <- breakpoints(formula, data = GermanM1)
rid <- recresid(formula, data = GermanM1)
exportJson <- toJSON(summary(bp), force=TRUE, pretty = TRUE)
write(exportJson, paste(PATH_OUT_MODEL_SPECS, "GermanM1_bp.json", sep="/"))
exportJson <- toJSON(rid, force=TRUE, pretty = TRUE)
write(exportJson, paste(PATH_OUT_MODEL_SPECS, "GermanM1_rid.json", sep="/"))
