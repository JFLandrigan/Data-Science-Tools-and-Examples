cv.nn <- function(dat = NA, inVars = NA, outVars = NA, hidLayers = NA, foldCol = NA, 
                  learn_rate = .01, lin_out = FALSE, thresh = .01, steps = 100000, alg = 'rprop+'){
  
  require(caret)
  require(neuralnet)
  
  #set up the model formula
  mod.formula <- as.formula(paste(paste(outVars, collapse = "+"), "~", paste(inVars, collapse = " + ")))
  
  #get the testing indeces using the createFolds function provided by the caret package
  folds <- createFolds(foldCol, k = 10)
  
  #results is a vector that will contain the accuracy for each of the network trainings and testing
  results <- c()
  
  for (fld in folds){
    
    #train the network (note I have subsetted out the indeces in the validation set)
    nn <- neuralnet(formula = mod.formula, data = dat[-fld,], hidden = hidLayers, algorithm = alg,
                    learningrate = learn_rate, linear.output = lin_out, threshold = thresh, stepmax = steps) 
    
    #get the classifications from the network
    preds <- compute(nn, dat[fld , inVars]) 
    
    if(lin_out){
      #Calc RMSE
      results <- c(results, sqrt(mean((preds$net.result - dat[fld, outVars])^2)))
  
    }else{
      #Calc the mean classification accuracy
      classRes <- predss$net.result
      nnClass <- apply(classRes, MARGIN = 1, which.max)
      origClass <- apply(dat[fld , outVars], MARGIN = 1, which.max)  
      results <- c(results, mean(nnClass == origClass) * 100)
    }
  } 
  
  #Train the network on the entire dataset
  nn <- neuralnet(formula = mod.formula, data = dat, hidden = hidLayers, algorithm = alg,
                  learningrate = learn_rate, linear.output = lin_out, threshold = thresh) 
  
  return(list(Acc = round(mean(results),2), loopAccs = results, finalNet = nn, modFormula = mod.formula))
  
}