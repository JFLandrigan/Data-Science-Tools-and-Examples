#Parcellation Script
#This script can perform a basic parcellation (i.e. calculate the percent damage in a region) for lesion scans.
#It depends on having the LESYMAP package installed (see: https://github.com/dorianps/LESYMAP)

#read in lesymap package
library(LESYMAP)

#read in the parcellation image (i.e. the image that contains the info for the white matter tracts) 
#just needs the file name and or path/filename
prcImg <- antsImageRead("")

#read in the lesions (this is a function that Dan created for reading in images)
#if working with cheaha there is a repository that you should be able to pull from
#regCodes is the base codes of the images
#set the image directory
imgDir <- ""
#get the file names from the folder storing the lesion files
flnames <- list.files(imgDir)
#initiate the list to store the images
lesions <- list()
for(i in 1:length(flnames)){
  lesions[[i]] <- antsImageRead(paste0(imgDir,flnames[i]))
}

#if the lesion files have different dimensions then the template then register the lesions to the template dimensions
for(i in 1:length(lesions)){
  #register the lesions to the new template
  trans <- antsRegistration(fixed = prcImg, moving = lesions[[i]], typeofTransform = c("Translation"))
  #replace the original lesion img with the new warped img to the same spot in the list
  lesions[[i]] <- trans$warpedmovout
}

#perform the parcellation
#The function returns a matrix with the number of rows equal to the number of scans and each col is a separate region 
#(each cell is the percent damage for the given region for the scan)
parcelMat <- getLesionLoad(lesions.list = lesions, parcellation = prcImg)

#If wanted you can rename the column names 
colnames(whiteMatParcel) <- paste0("region", colnames(parcelMat))

