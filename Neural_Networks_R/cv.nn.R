cv.nn <- function(dat = NA, inVars = NA, output = NA, hidLayers = NA, numFolds = 10, 
                  learn_rate = .01, lin_out = FALSE, thresh = .01, steps = 100000, alg = 'rprop+'){
  
  #The function requires the caret and neuralnet packages to be used.
  #dat should be a dataframe containing all the input and output data
  #inVars should be a vector containing the names of the columns for the input variables
  #output arg should be the name of the column containing the output values
  #hidLayers expects a vector containg the numerical values for the number of units in each hidden layer 
  #The alg argument can accept any of the algorithms supported by neuralnet()
  
  #load required packages
  require(caret)
  require(neuralnet)
  
  if(lin_out == FALSE){
    #generate the one hot encoding matrix
    outMat <- model.matrix(~ 0 + dat[,output], dat)
    #change the names of the outMat and store the names of the outcome variables
    colnames(outMat) <- unique(dat[,output])
    outnames <- colnames(outMat)
    
    #bind the one hot encoding matrix to the dat
    dat <- cbind(dat, outMat)
  }else{
    outnames <- output
    }
  
  #set up the model formula
  mod.formula <- as.formula(paste(paste(outnames, collapse = "+"), "~", paste(inVars, collapse = " + ")))
  
  #If classification set up the missClassifcation matrix else keep as NA
  if(lin_out == FALSE){
    #store misclass info
    numClasses <- length(unique(dat[,output]))
    missClass <- array(data = rep(0, numClasses^2), dim = c(numClasses, numClasses))
  }else{
    missClass <- NA
  }
  
  #get the testing indeces using the createFolds function provided by the caret package
  folds <- createFolds(dat[,output], k = numFolds)
  
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
      results <- c(results, sqrt(mean((preds$net.result - dat[fld, outnames])^2)))
  
    }else{
      #Calc the mean classification accuracy
      classRes <- preds$net.result
      nnClass <- apply(classRes, MARGIN = 1, which.max)
      origClass <- apply(dat[fld , outnames], MARGIN = 1, which.max)  
      results <- c(results, mean(nnClass == origClass) * 100)
      
      #Update the missclass matrix
      for(class in 1:length(outnames)){
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
    rownames(missClass) <- outnames
    colnames(missClass) <- outnames
  }
  
  #Train the network on the entire dataset
  nn <- neuralnet(formula = mod.formula, data = dat, hidden = hidLayers, algorithm = alg,
                  learningrate = learn_rate, linear.output = lin_out, threshold = thresh) 
  
  return(list(Acc = round(mean(results),2), foldAccs = results, finalNet = nn, modFormula = mod.formula, classMat = missClass))
  
}
