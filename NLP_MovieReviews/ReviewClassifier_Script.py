#Import the packages needed for the analysis
import os
import nltk
from nltk.tokenize import word_tokenize
import random
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 

#get_data() takes a working directory and and type
#the function reads in the data from a file and then stores the text in a tuple along 
#with the type (pos or neg) 
def get_data(direct = os.getcwd(), tp = 'neg'):
    #get the files listed in the directory
    fls = os.listdir(direct)
    
    revs = []
    for i in range(len(fls)):
        #Read in the file contents
        fl = open(direct+fls[i],"r") 
        txt = fl.read()
        #generate a list of words by tokenizing the document
        wrds = word_tokenize(txt)
        #append the type of the 
        revs.append((wrds,tp))
        #Close the file connection
        fl.close()        
    
    return(revs)
    
#Call the get_data function which returns lists of tuples (each tuple is vec of words and type)
posData = get_data(direct = 'movie_reviews/pos/', tp = "pos")
negData = get_data(direct = 'movie_reviews/neg/', tp = "neg")

#Combine the positive and negative review lists
all_revs = posData + negData
#shuffle the reviews 
random.shuffle(all_revs)

#Append all the words to a single vector
words = []
for i in range(len(all_revs)-1):
    words = words + all_revs[i][0]
#put all the words in lowercase so that don't get duplicates (i.e. Sit / sit) when 
#grabbing the unique set of words
words = [w.lower() for w in words]

#Generate a frequency distribution of all the words in the reviews
words = nltk.FreqDist(words)

#Get the top 4000 words for the features to be used
features = list(words.keys())[:4000]

#Determine if the feature words are in a given review (create boolean feature set)
rev_fts = []
#loop through all the reviews
for rev in all_revs:
    #generate the set of unique words in the review
    txt_set = set(rev[0])
    fts = {}
    #loop through each of the words in the feature set
    for w in features:
        #determine if the word is in the review
        fts[w] = (w in txt_set)
    #store the results in the rev_fts list
    rev_fts = rev_fts + [(fts,rev[1])]


#kf_classifier() is function to perform k-fold cross validation for the classifiers
def kf_classifier(classi, fts, k = 10):
    #import the packages needed for the function
    from numpy import array
    from sklearn.cross_validation import KFold
    
    #Generate cv sets of training and testing indeces
    kf = KFold(len(fts), n_folds = 10, shuffle = False)
    
    accs = []
    # print the contents of each training and testing set
    for iteration, data in enumerate(kf, start=1):
        #Grab the training and testing indeces
        train = [fts[x] for x in data[0]]
        test  = [fts[x] for x in data[1]]
        #Train the classifier
        classi.train(train)
        #Calculate the percent correct classification and store it in accs
        accs.append(nltk.classify.accuracy(classi, test)*100)
        print("Completed Fold: " + str(iteration))
    
    #return the accuracies for the folds
    return(array(accs))

#Test the algorithms using kf_classifier()

#Logistic Regression
log_accs = kf_classifier(classi = SklearnClassifier(LogisticRegression()), fts = rev_fts, k = 10)
print("Logistic Regression Mean Accuracy: ", log_accs.mean().round())
#SVC
svc_accs = kf_classifier(classi = SklearnClassifier(SVC()), fts = rev_fts, k = 10)
print("SVC Mean Accuracy: ", svc_accs.mean().round())
#MultiNomial_NaiveBayes
nb_accs = kf_classifier(classi = SklearnClassifier(MultinomialNB()), fts = rev_fts, k = 10)
print("Naive Bayes Mean Accuracy: ", nb_accs.mean().round())
