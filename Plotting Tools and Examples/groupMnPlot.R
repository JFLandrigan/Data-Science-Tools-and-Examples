groupMnPlot <- function(dat = NA, groupingCol = NA, measureVars = NA, scale = FALSE, horizontal = TRUE, groupColors = "Set1", rotateXLab = FALSE, groupLabel = "", xlabel = "", ylabel =""){
  
  #clustPlot function used for plotting mean measures for clusters of 
  #dat - expects a dataframe in wide format i.e. each row is a single observation and the columns are the measurement dimensions
  #groupingCol - name of the column containing the group levels
  #measureVars - names of the columns containing the measurements for the groups
  #scale - default is FALSE if set to true then measureVars are scaled
  #horizontal - default is TRUE, flipping the coordinates of the plot
  #groupColors - palette colors (options are those available for scale_fill_brewer)
  #rotateXLab - if set to TRUE then rotates the x axis tick labels
  
  #load ggplot
  require(ggplot2)
  #load reshape
  require(reshape)
  #load doby 
  require(doBy)
  
  #if scale is true then perform scaling on the measurement columns
  if(scale){
    dat[,measureVars] <- scale(dat[,measureVars])
  }
  
  #generate table that has mean measure values by each cluster
  mf <- as.formula(paste(paste(measureVars, collapse = "+"), "~", groupingCol))
  mnDat <- as.data.frame(t(summaryBy(mf, data = dat, FUN = c(mean), keep.names = TRUE, na.rm = TRUE)))[2:(length(measureVars)+1),]
  colnames(mnDat) <- unique(dat[,groupingCol])
  #add the measures to the mean dataframe
  mnDat$measures <- row.names(mnDat)
  
  #convert the data into long format 
  loadings.dat <- melt(mnDat, id.vars = "measures", 
                       measure.vars = unique(dat[,groupingCol]), 
                       variable_name = "Groups")
  #convert the value column back to numeric values
  loadings.dat$value <- as.numeric(as.character(loadings.dat$value))
  
  
  #Define the rectangles to be plotted in the background og the plot
  rectangles <- data.frame(
    xmin = seq(1.5, length(measureVars)+.5, by=2) - 1,
    xmax = seq(1.5, length(measureVars)+.5, by=2),
    ymin = -Inf,
    ymax = Inf
  )
  
  #generate the plot
  p <- ggplot() +
    #add the rectangles to the background
    geom_rect(data = rectangles, aes(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax), fill='gray80', alpha=0.8) +
    #add in the loadings data
    geom_bar(data = loadings.dat, aes(x = measures, y = value, fill = Groups), 
             stat = "identity",position = "dodge") +
    scale_x_discrete(limits = levels(loadings.dat$measures)) +
    scale_fill_brewer(palette = groupColors, name = groupLabel) +
    geom_hline(yintercept = 0, colour = "black") +
    #geom_vline(xintercept = seq(1.5, length(measureVars) + .5, by = 1)) +
    xlab(xlabel) +
    ylab(ylabel) +
    theme_bw(base_size=12) + 
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())
  
  #if the horizontal flag is TRUE then flip the plot onto its side
  if(horizontal){
    p <- p + coord_flip()
  }
  #if rotateXlab then rotates the x axis labels 90 degrees
  if(rotateXLab){
    p <- p + theme(axis.text.x = element_text(angle = 90, hjust = 1))
  }
  
  #return the plot object
  return(p)
  
}
