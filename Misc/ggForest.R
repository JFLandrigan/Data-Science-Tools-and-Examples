ggForest <- function(metaObj = NA, labs = NA, ES_Text = TRUE){
  
  #expects a meta-analysis object returned from the metaFor rma.uni
  #labs: vector of study names note it should be in the order of the studys in the metaobj
  #ES_Text: logical whether or not to add the effect size and standard error to the plot
  
  #load likert package for reverse.levels functions
  require(likert)
  #load ggplot
  require(ggplot2)
  #load gridExtra for arranging plots when text wanted
  require(gridExtra)
  #load stringr for the padding function when printing the text
  require(stringr)
  
  #If no labels passed in use the labels from the meta object else use the supplied labels
  if(is.na(labs)){
    labs <- metaObj$slab
  }
  
  #Construct a dataframe with the relevant information for plotting
  df <- data.frame(Study = labs, EffectSizes = metaObj$yi, SE = metaObj$vi, 
                   Type = rep("Reference",length(labs)))
  
  #Add the overall effect to the dataframe
  df <- rbind(df, data.frame(Study = "Overall", EffectSizes = metaObj$beta[1], SE = metaObj$se[1],
                             Type = "Overall SMD"))
  
  #calculate the 95% cis
  df$lower_ci = (-1.96 * df$SE) + df$EffectSizes
  df$upper_ci = (1.96 * df$SE) + df$EffectSizes
  
  #reverse the order of the studies for plot
  df$Study <- reverse.levels(df$Study)
  
  #Generate the forest plot
  f <- ggplot() + geom_point(data = df, aes(x = EffectSizes, y = Study), colour = "black") + 
    geom_errorbarh(data = df, aes(x = EffectSizes, y = Study, xmin = lower_ci, xmax = upper_ci),height = .1) +
    xlab('SMD') +
    ylab('Reference') +
    #Add a vertical dashed line indicating an effect size of zero and the overall effect, for reference
    geom_vline(xintercept = 0, color = 'black', linetype = 'dashed') + 
    geom_vline(xintercept = metaObj$beta[1], linetype = 'dashed', color = 'red') +
    #Break the plot into two grids so that the overall effect appears below the references
    facet_grid(Type ~., scales = 'free', space = 'free') + 
    theme_bw() +
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), 
          strip.background = element_blank(), strip.text.y = element_blank(), 
          #axis.title.x = element_blank(), 
          legend.position = "none")
  
  #if ES_Text is TRUE create a plot with the effect sizes and return both plots as grid obj
  if(ES_Text){
    #Create a column with the ES and CI pasted together
    df$ES_CI_Txt <- paste(str_pad(sprintf("%.2f", round(df$EffectSizes,2)), 3), " ", 
                          paste0("[", str_pad(sprintf("%.2f",round(df$lower_ci,2)),3), ", ", 
                                 str_pad(sprintf("%.2f",round(df$upper_ci,2)),3), "]"))
    
    t <- ggplot(df, aes(x = rep("SMD [95% CI]", length(Study)), y = Study, label = ES_CI_Txt)) +
      geom_text() +
      xlab("") +
      facet_grid(Type ~ ., scales = 'free', space = 'free') +
      theme_bw() +
      theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), 
            strip.background = element_blank(), strip.text.y = element_blank(), 
            axis.ticks.x = element_line(colour = "transparent"),
            #Strip all the y axis info
            axis.title.y = element_blank(),
            axis.text.y = element_blank(),
            axis.ticks.y = element_blank(),
            #remove any legends
            legend.position = "none")
    
    return(grid.arrange(f, t, ncol = 2, widths = c(4,1.5)))
  }else{
    return(f)
  }
}
