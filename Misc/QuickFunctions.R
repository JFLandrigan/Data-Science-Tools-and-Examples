#Quick functions

#perform majority vote 
#dat should be dataframe or matrix where rows = obs and cols = the classifications from
#dif analyses
majorityVote <- function(dat = NA, thresh = .5){
  #calc num instances obs needs to be classified
  classThresh <- dim(dat)[2] * thresh
  mv <- c()
  for(i in 1:dim(dat)[1]){
    cnts <- data.frame(table(t(dat[i,])))
    if(max(cnts$Freq) >= classThresh){
      mv <- c(mv, as.character(cnts$Var1[which.max(cnts$Freq)]))
    }else{
      mv <- c(mv, NA)
    }
  }
  return(mv)
}

#perform a median split
medSplit <- function(x){return(ifelse(x > median(x),1,0))}

#perform a mean split
mnSplit <- function(x){return(ifelse(x > mean(x),1,0))}

#calc percent missing data
percMiss <- function(x){return(mean(is.na(x)))}

#standardize the values to be between 0 and 1
stdzVals <- function(x){return( (x - min(x)) / (max(x) - min(x)) )}

