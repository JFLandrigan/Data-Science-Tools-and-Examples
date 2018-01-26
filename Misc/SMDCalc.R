SMDCalc <- function(mn_exp = NA, sd_exp = NA, n_exp = NA, mn_ctl = NA, sd_ctl= NA, n_ctl = NA, hedges = TRUE ){
  
  #Function to calculate Standardized Mean Differences 
  #can either take single values or vectors of values
  #returns list with effect sizes and standard errors
  
  #calc within group standard dev pooled across groups
  s_within <- sqrt( (((n_exp-1)*sd_exp^2) + ((n_ctl-1)*sd_ctl^2)) / (n_exp+n_ctl-2) )
  #calc standardized mean dif
  d <- (mn_exp-mn_ctl) / s_within
  #calc variance of d
  v <- ((n_exp+n_ctl)/(n_exp*n_ctl)) + (d^2/(2*(n_exp+n_ctl)))
  #calc the standard error of d
  se <- sqrt(v)
  
  if(hedges == FALSE){
    return(list(EF = d, SE = se))
  }else{
    #bias correction with hedges
    j = 1 - (3 / (4*(n_exp+n_ctl-2)-1) )
    #calculate hedges g
    g <- j * d
    #calc variance of g
    vg <- j^2 * v
    #calc standard error of g
    seg <- sqrt(vg)
    
    return(list(EF = g, SE = seg))
  }
}
