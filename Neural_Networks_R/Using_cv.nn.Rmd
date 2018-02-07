---
title: "Using cv.nn.R"
author: "Jon-Frederick Landrigan"
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

Source the function from github so that it is in the environment
```{r}
source("https://raw.githubusercontent.com/JFLandrigan/Data-Science-Tools-and-Examples/master/Neural_Networks_R/cv.nn.R")
```

First I will use the function for a classification problem. I will be using the iris dataset which contains the petal and sepal lengths and widths for three different iris species.
```{r}
irisDat <- iris
```

Now that I have loaded in the data I can call the function to perform the analysis. Note by default the function performs 10 fold cross validation.
```{r,message=FALSE,warning=FALSE}
res <- cv.nn(dat = irisDat, inVars = colnames(irisDat)[1:4], output = "Species", hidLayers = c(4,3,2))
```
The mean classification accuracy of the network was `r res$Acc`%

Now I will take a look at the classification information. 
```{r}
res$classMat
```

For the next example I will be using the neuralnet to perform a regression. The data that will be used is the Boston dataset from the MASS package.
```{r,message=FALSE,warning=FALSE}
library(MASS)

dat <- Boston
```

First I will scale the data.
```{r}
maxVals <- apply(dat, 2, max) 
minVals <- apply(dat, 2, min)

scaled <- as.data.frame(scale(dat, center = minVals, scale = maxVals - minVals))
```

Now I will train and test the network on the scaled data. 
```{r}
res <- cv.nn(dat = scaled, inVars = colnames(scaled)[1:13], output = "medv", hidLayers = c(5,3), 
             lin_out = TRUE, learn_rate = NULL)
```
The root mean squared error of the network was `r res$Acc`
