ggBiasFigs <- function(dat = NA, grid = TRUE, bar = TRUE,  sumFig = TRUE){
  
  #ggBiasFig is a function to make basic figures to display bias information
  #dat - expects a dataframe in wide format with a column called Study containing the study labels and
  #   the bias measures in the subsequent columns (currently the cells should be -1 = high risk, 0 = uncertain,
  #   1 = low risk) These are based on the cochrane guidelines
  #grid - logical indicating whether or not to generate a grid plot
  #bar - logical indicating whether or not to generate a percentage bar plot
  #returns a list containing the desired plots
  
  #load in ggplot, likert and reshape package
  require(ggplot2)
  require(reshape)
  require(likert)
  
  #convert the wide dat to a long dataframe
  longDat <- melt(dat, 
                  id.vars = "Study", 
                  measure.vars = colnames(dat)[2:dim(dat)[2]])
  #Clean up the variable names (col names with space are read in with .)
  longDat$variable <- gsub(".", replacement = " ", longDat$variable, fixed = TRUE)
  
  
  if(grid){
  
    gr <- ggplot(longDat, aes(Study, variable, fill = as.character(value))) + 
      geom_tile(aes(width = 0.95, height = 0.95)) + 
      #set colors
      scale_fill_manual(name = "Bias",
                        values = c("-1" = "red", "0" = "yellow", "1" = "green"), 
                        labels = c("High", "Uncertain", "Low")) +
      #black and white theme with set font size
      theme_bw(base_size = 10) + 
      #rotate x-axis labels so they don't overlap, get rid of unnecessary axis titles, adjust plot margins
      theme(axis.text.x = element_text(angle = 90, size = 10),
            axis.text.y = element_text(size = 10),
            axis.title.x = element_blank(), 
            axis.title.y = element_blank(),
            legend.text=element_text(size=10)) # + guides(fill = F)
  
  }else{gr <- NA}
  
  if(bar){
    
    #reverse the order of the studies for plot
    longDat$Study <- reverse.levels(longDat$Study)
    
    scaleFact <- length(unique(longDat$Study))
    
    #-1 = red, 0 = green, 1 = blue ---ggplot default
    
    br <- ggplot(longDat, aes(Study, fill = as.character(value))) +
      geom_bar(aes( y = ((..count..)/sum(..count..))*20 )) + 
      scale_y_continuous(labels = scales::percent) +
      scale_fill_manual(name = "Bias",
                        values = c("-1" = "red", "0" = "yellow", "1" = "green"), 
                        labels = c("High", "Uncertain", "Low")) +
      #flip the coordinates so studies along y
      coord_flip() +
      theme_bw(base_size = 10) +
      theme(axis.text.x = element_text(angle = 90, size = 10),
            axis.text.y = element_text(size = 10), 
            axis.title.x = element_blank(), 
            axis.title.y = element_blank(),
            legend.text=element_text(size=10)) #+ guides(fill = T)
    
  }else{br <- NA}
  
  if(sumFig){
    
    scaleFact <- length(unique(longDat$variable))
    
    sf <- ggplot(longDat, aes(variable, fill = as.character(value))) +
      geom_bar(aes( y = ((..count..)/sum(..count..))*7 )) + 
      scale_y_continuous(labels = scales::percent) +
      scale_fill_manual(name = "Bias",
                        values = c("-1" = "red", "0" = "yellow", "1" = "green"), 
                        labels = c("High", "Uncertain", "Low")) +
      #flip the coordinates so studies along y
      coord_flip() +
      theme_bw(base_size = 10) +
      theme(axis.text.x = element_text(angle = 90, size = 10),
            axis.text.y = element_text(size = 10), 
            axis.title.x = element_blank(), 
            axis.title.y = element_blank(),
            legend.text=element_text(size=10))
    
  }else{sf < NA}
  
  return(list(grid = gr, percBars = br, biasSummary = sf))
  
}
