cv.nn <- function(dat = NA, inVars = NA, outVars = NA, hidLayers = NA, foldCol = NA, numFolds = 10, 
                  learn_rate = .01, lin_out = FALSE, thresh = .01, steps = 100000, alg = 'rprop+'){
  
  #The function requires the caret and neuralnet packages to be used.
  #The alg argument can accept any of the algorithms supported by neuralnet()
  #outVars and inVars should be vectors containing the names of inputs (inVars) and the output(s) (outVars)
  #For classification neuralnet() expects the outcome var to be coded using the one hot encoding method
  
  #load required packages
  require(caret)
  require(neuralnet)
  
  #set up the model formula
  mod.formula <- as.formula(paste(paste(outVars, collapse = "+"), "~", paste(inVars, collapse = " + ")))
  
  #get the testing indeces using the createFolds function provided by the caret package
  folds <- createFolds(foldCol, k = numFolds)
  
  #results is a vector that will contain the accuracy for each of the network trainings and testing
  results <- c()
  
  #Run cross validation on the network
  for (fld in folds){
    
    #train the network 
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
  
  return(list(Acc = round(mean(results),2), foldAccs = results, finalNet = nn, modFormula = mod.formula))
  
}
