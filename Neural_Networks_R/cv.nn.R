cv.nn <- function(dat = NA, inVars = NA, outVars = NA, hidLayers = NA, foldCol = NA, numFolds = 10, 
                  learn_rate = .01, lin_out = FALSE, thresh = .01, steps = 100000, alg = 'rprop+'){
  
  #Function to run k-fold cross validation with neuralnet
  #The function requires the caret and neuralnet packages to be used.
  #The alg argument can accept any of the algorithms supported by neuralnet()
  #outVars and inVars should be vectors containing the names of inputs (inVars) and the output(s) (outVars)
  #For classification neuralnet() expects the outcome var to be coded using the one hot encoding method
  #foldCol arg should be single column that contains the true outputs 
  
  #load required packages
  require(caret)
  require(neuralnet)
  require(plyr)
  
  #set up the model formula
  mod.formula <- as.formula(paste(paste(outVars, collapse = "+"), "~", paste(inVars, collapse = " + ")))
  
  #If classification set up the missClassifcation matrix else keep as NA
  if(lin_out == FALSE){
    #store misclass info
    numClasses <- length(unique(foldCol))
    missClass <- array(data = rep(0, numClasses^2), dim = c(numClasses, numClasses))
  }else{
    missClass <- NA
  }
  
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
      classRes <- preds$net.result
      nnClass <- apply(classRes, MARGIN = 1, which.max)
      origClass <- apply(dat[fld , outVars], MARGIN = 1, which.max)  
      results <- c(results, mean(nnClass == origClass) * 100)
      
      #Update the missclass matrix
      for(class in 1:length(outVars)){
        #get the indeces for where the origClass was predicted in the fold by nn
        inds <- which(origClass == class)
        #get the count info for number of classifications of each type by nn
        missinfo <- count(nnClass[inds])
        #update the missClass matrix according to the row index (class) and the col indeces (classified as)
        missClass[class,missinfo$x] <- missClass[class,missinfo$x] + missinfo$freq 
      }
    }
  } 
  
  #if classification add column and row names to the miss class matrix
  if(lin_out == FALSE){
    rownames(missClass) <- outVars
    colnames(missClass) <- outVars
  }
  
  #Train the network on the entire dataset
  nn <- neuralnet(formula = mod.formula, data = dat, hidden = hidLayers, algorithm = alg,
                  learningrate = learn_rate, linear.output = lin_out, threshold = thresh) 
  
  return(list(Acc = round(mean(results),2), foldAccs = results, finalNet = nn, modFormula = mod.formula, classMat = missClass))
  
}
