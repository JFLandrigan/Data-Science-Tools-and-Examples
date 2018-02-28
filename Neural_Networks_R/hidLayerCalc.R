hidLayerCalc <- function(numInputUnits = NA, numOutputUnits = NA, dropSize = .66, addOne = FALSE, initLayer = FALSE){
  #This function returns a vector of numbers that is compatible with the neuralnet function for the hidden layers.
  #numInputUnits - the number of input features for the network
  #numOutputUnits - number of possible outputs for the neural net
  #dropSize - how many units to drop in each layer
  #initLayer - if true then the first hidden layer has the same number of units as the input layer
  
  check <- FALSE
  ct <- 1
  hidNums <- c()
  
  #go through while loop dropping the size of the hidden layers by dropsize 
  while(check == FALSE){
    #if this is the first time through the loop mult the num input by dropsize
    if(ct == 1){
      if(initLayer){
        hidNums <- c(hidNums, numInputUnits)
      }else{
      hidNums <- c(hidNums, round(numInputUnits * dropSize))
      }
    #if this an iteration beyond the first then calculate the new layer by mult of numunits in previous layer by dropsize
    }else{
      hidNums <- c(hidNums, round(hidNums[ct-1] * dropSize))
    }
    #if the number of hidden units in the last layer is less then or equal to the number of units in the output layer then
    #stop adding layers and remove the final layer
    if(hidNums[ct] <= numOutputUnits){
      hidNums <- hidNums[-length(hidNums)]
      check <- TRUE
    }
    ct = ct + 1
  }
  
  #Add a final hidden layer with a single unit if addOne is True
  if(addOne == TRUE){
    hidNums <- c(hidNums, 1)
  }
  #Return the vector of hidden units
  return(hidNums)
}
