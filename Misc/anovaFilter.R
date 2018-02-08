anovaFilter <- function(dat = NA, groups = NA, measureVars = NA, pThresh = .05){
  
  #dat - expects a dataframe or matrix object
  #groups - name of the column containing the grouping values
  #measureVars - variables to use as the outcome of the anova
  #pThresh - value used to threshold the measure vars default of .05
  
  #initialize the p and f val vectors
  f <- c()
  p <- c()
  
  #run an anova on each of the measure vars against the outcomes
  for(vari in measureVars){
    #define the model formula
    modform <- as.formula(paste(vari, "~", groups))
    #run an anova and store the summary info
    x <- summary(aov(modform, data = dat))
    f <- c(f, x[[1]]$F[1])
    p <- c(p, x[[1]]['Pr(>F)'][[1]][1])
  }
  
  #generate a dataframe that is subset according to the measureVars where there were significant differences
  cols <- colnames(dat)[!colnames(dat) %in% measureVars]
  cols <- c(cols, measureVars[p < pThresh])
  subdat <- subset(dat, select = cols)
  
  #return the filtered dataframe, the vector of f vals and vector of pvals in list object
  return(list(filtDat = subdat, fVals = f, pVals = p))
}