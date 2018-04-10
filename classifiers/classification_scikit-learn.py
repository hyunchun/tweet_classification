import pandas as pd
import numpy as np
import itertools
import string
import sys
import operator 

try:
    import json
except ImportError:
    import simplejson as json

from sklearn.svm import SVC, LinearSVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn import metrics

from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn.utils import resample

from sklearn import metrics


def extract_from_json(inFile):
	textSet = []
	contentLabelSet = []
	typeLabelSet = []

	for line in inFile:
		line = json.loads(line)
		textSet.append(line["text"])
		contentLabelSet.append(int(line["content_label"]))
		typeLabelSet.append(int(line["type_label"]))

	return textSet, contentLabelSet, typeLabelSet

def label_separator(label_to_separate, text_set, content_label_set):
	majority, majority_label, minority, minority_label = []
	if label_to_separate == "content":
		count = 0
		for line in text_set:
			if content_label_set[count] == 6:
				majority.append(text_set[count])
				majority_label.append(content_label_set[count])
			else: 
				minority.append(text_set[count])
				minority_label.append(content_label_set[count])

	return majority, majority_label, minority, minority_label

def extract_dictionary(dataset, word_dict):
    index = 0
    for line in dataset:
    	line = line.split()
    	for word in line:
    		# just word itself 
    		if word not in word_dict:
    			word_dict[word] = index
    			index += 1
                try:
    		    # lower capiticalization of the word
    		    lower_word = str(word).lower()
    	            if lower_word not in word_dict:
    	    	        word_dict[lower_word] = index
                        index += 1

    		    # no punctuation 
	            replace_punctuation = str(word).maketrans(string.punctuation, ' ' * len(string.punctuation))
	            clean_word = str(word).translate(replace_punctuation)

	            if clean_word not in word_dict:
                        word_dict[clean_word] = index
                        index += 1
                except:
                    continue

    print("done with extract_dictionary")

    return word_dict

def generate_feature_matrix(dataset, word_dict):
    number_of_tweets = len(dataset)
    number_of_words = len(word_dict)

    feature_matrix = np.zeros((number_of_tweets, (number_of_words + 3)))

    train_dict = {}
    line_count = 0
    for line in dataset:
    	line = line.split()
    	for word in line:
    		# check if RT
    		if word == "RT":
    			feature_matrix[line_count][number_of_words] += 1
    			break

    		# check if hashtag
    		if word[0] == "#":
    			feature_matrix[line_count][number_of_words + 1] += 1

    		# check if mention
    		if word[0] == "@":
    			feature_matrix[line_count][number_of_words + 2] += 1

    		# just word itself 
    		feature_matrix[line_count][word_dict[word]] += 1

                try:
    		    # lower capiticalization of the word
    		    lower_word = str(word).lower()
                    feature_matrix[line_count][word_dict[lower_word]] += 1
    
    	            # no punctuation 
	            replace_punctuation = str(word).maketrans(string.punctuation, ' ' * len(string.punctuation))
        	    clean_word = str(word).translate(replace_punctuation)

                    feature_matrix[line_count][word_dict[clean_word]] += 1
                except:
                    continue
        line_count += 1

    print("done with feature_matrix")

    return feature_matrix

def cv_performance(clf, X, y, k=2, metric="accuracy"):
    scores = []

    skf = StratifiedKFold(n_splits=k, shuffle=True)

    for train_index, test_index in skf.split(X, y):
        #print(X.shape)
        #print(y.shape)
        #print("X: ", X)
        #print("y: ", y)
        #print("train_index: ", train_index)
        #print("test_index: ", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # accuracy
        if (metric=="accuracy"):
            score = accuracy_score(y_test, y_pred)
            scores.append(score)
        # F1-score
        elif (metric=="f1-score"):
            scores.append(f1_score(y_test, y_pred))
            
    # And return the average performance across all fold splits.
    return np.array(scores).mean()

# ----------------------------------------------------------

def main():
    train_file = open("%s" %(sys.argv[1]))
    # test_file = open("%s" %(sys.argv[2]))
    train_text_set, train_content_label_set, train_type_label_set = extract_from_json(train_file)
    # test_text_set, test_content_label_set, test_type_label_set = extract_from_json(test_file)
    train_content_label = np.asarray(train_content_label_set)
    train_type_label = np.asarray(train_type_label_set)
    # content
    #majority, majority_label, minority, minority_label = label_separator("content", train_text_set, train_content_label_set)

    # resample minority and majority classes
    #majority_dwsampled, majority_dwsampled_label = resample(majority, majority_label, replace=False, n_samples=int(len(minority)/3), random_state=123)

    # resampled_train_text_set = minority + majority_dwsampled
    # resampled_train_content_label_set = minority_label + majority_dwsampled_label
	
    word_dict = {}
    word_dict = extract_dictionary(train_text_set, word_dict)
    # word_dict = extract_dictionary(test_text_set, word_dict)

    train_feature_matrix = generate_feature_matrix(train_text_set, word_dict)
    # test_feature_matrix = generate_feature_matrix(test_text_set, word_dict)

    c_range = [10**(-1), 10**(0), 10**(1), 10**(2)]
    #c_range = [10**(-1), 10**(0), 10**(1), 10**(2)]
    
    #print(train_feature_matrix)
    print("OneVsRestClassifier: content_label")

    print("\tlinear SVC")
    for c in c_range:
        #c = np.random.uniform(-1, 1 )
        #c = 10 ** c
        #print(c)
    	svc_i = OneVsRestClassifier(SVC(kernel = 'linear', C = c, class_weight = 'balanced'))
	score = cv_performance(svc_i, train_feature_matrix, train_type_label, 2, "accuracy")
	print("\t\tcurrent c: ", c, ", performance: ", score)
   
    """
    print("\tlinear SVC with l1 loss")
    for c in c_range:
	svc_i = OneVsRestClassifier(LinearSVC(penalty='l1', loss='squared_hinge', dual=False, C=c, class_weight='balanced'))
        score = cv_performance(svc_i, train_feature_matrix, train_type_label, 5, "accuracy")
        print("\t\tcurrent c: ", c, ", performance: ", score)
    
    print("\tquadratic SVC")
    for i in range(0, 5):
        current_coef0 = np.random.uniform(-3, 3)
       	print(i, "coef0: ", current_coef0)
	for c in c_range:
	    svc_i = OneVsRestClassifier(SVC(kernel='poly', degree=2, C=c, coef0=current_coef0, class_weight='balanced'))
	    score = cv_performance(svc_i, train_feature_matrix, train_content_label, 2, "accuracy")
	    print("\t\tcurrent c: ", c, ", current coef0: ", current_coef0, ", performance: ", score)
    """
    
    print("OneVsOneClassifier: content_label")
    
    print("\tlinear SVC")
    for c in c_range:
    	svc_i = OneVsOneClassifier(SVC(kernel = 'linear', C = c, class_weight = 'balanced'))
	score = cv_performance(svc_i, train_feature_matrix, train_content_label, 2, "accuracy")
	print("\t\tcurrent c: ", c, ", performance: ", score)
    """
    print(" \tlinear SVC with l1 loss")
    for c in c_range:
	svc_i = OneVsOneClassifier(LinearSVC(penalty='l1', loss='squared_hinge', dual=False, C=c, class_weight='balanced'))
	score = cv_performance(svc_i, train_feature_matrix, train_content_label, 2, "accuracy")
	print("\t\tcurrent c: ", c, ", performance: ", score)
    
    print("\tquadratic SVC")
    train_set_q = train_feature_matrix[0:1900]
    train_set_q_l = train_content_label[0:1900]
    test_set_q = train_feature_matrix[1901:]
    test_set_q_l = train_content_label[1901:]
    for i in range(0, 5):
	current_coef0 = np.random.uniform(-3, 3)
	print(i, "coef0: ", current_coef0)
	for c in c_range:
	    svc_i = OneVsOneClassifier(SVC(kernel='poly', degree=2, C=c, coef0=current_coef0, class_weight='balanced'))
            #score = cv_performance(svc_i, train_feature_matrix, train_content_label, 2, "accuracy")
            svc_i.fit(train_set_q, train_set_q_l)
            y_pred = svc_i.predict(test_set_q)
            score = accuracy_score(test_set_q_l, y_pred)
	    print("\t\tcurrent c: ", c, ", current coef0: ", current_coef0, ", performance: ", score)
            if score < 0.35:
                break
    """
# ----------------------------------------------------------

if __name__ == '__main__':
    main()
