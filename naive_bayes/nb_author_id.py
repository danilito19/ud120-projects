#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 1 (Naive Bayes) mini-project 

    use a Naive Bayes Classifier to identify emails by their authors

    The NB algorithm will use prior probability to classify emails to their authors.
    
    authors and labels:
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

def nb(features_train, features_test, labels_train, labels_test):
	from sklearn.naive_bayes import GaussianNB
	#first create the classifier
	classifier = GaussianNB()

	#train algorithm and measure how long it takes
	t0 = time()
	classifier.fit(features_train, labels_train)
	print "training time:", round(time()-t0, 3), "s"

	#test algorithm and measure how long it takes
	t1 = time()
	predictions = classifier.predict(features_test)
	print "testing time:", round(time()-t1, 3), "s"

	#evaluate accuracy of predictions
	from sklearn.metrics import accuracy_score
	#compares my predicted labels to the testing labels
	print "the accuracy of the classifier is: " + str(accuracy_score(predictions, labels_test)) 
		#"which means that it was able to correctly classify that percentage of emails to the author"


nb(features_train, features_test, labels_train, labels_test)




