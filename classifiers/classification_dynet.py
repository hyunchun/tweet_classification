from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import itertools
import string
import sys
import operator 
import dynet as dy

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
    unique_content = []
    unique_type = []

    for line in inFile:
        line = json.loads(line.strip())
        textSet.append(line["text"].split())
        contentLabelSet.append(int(line["content_label"]))
        if int(line["content_label"]) not in unique_content:
            unique_content.append(int(line["content_label"]))
        typeLabelSet.append(int(line["type_label"]))
        if int(line["type_label"]) not in unique_type:
            unique_type.append(int(line["type_label"]))

    return textSet, contentLabelSet, typeLabelSet, unique_content, unique_type

def label_separator(label_to_separate, text_set, content_label_set, type_label_set):
    majority = []
    majority_content_label = []
    majority_type_label = []
    minority = []
    minority_content_label = []
    minority_type_label = []
    if label_to_separate == "type":
        count = 0
        for line in text_set:
            if int(type_label_set[count]) == 0:
                #print(text_set[count])
                majority.append(line)
                majority_content_label.append(content_label_set[count])
                majority_type_label.append(type_label_set[count])
            else: 
                minority.append(line)
                minority_content_label.append(content_label_set[count])
                minority_type_label.append(type_label_set[count])
            count += 1
    #majority_n = np.concatenate(majority)
    print(len(text_set))
    print(len(majority))
    print(len(minority))
    return majority, majority_content_label, majority_type_label, minority, minority_content_label, minority_type_label

def extract_dictionary(dataset, word_dict):
    index = 0
    for line in dataset:
        # line = line.split()
        #print(line)
        for word in line:
            #print(word)
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
                replace_punctuation = str(word).maketrans(string.punctuation, '')
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
        # line = line.split()
        for word in line:
            # check if RT
            if word == "RT":
                feature_matrix[line_count][number_of_words] += 1
                continue

            # check if hashtag
            if word[0] == "#":
                feature_matrix[line_count][number_of_words + 1] += 1

            # check if mention
            if word[0] == "@":
                feature_matrix[line_count][number_of_words + 2] += 1

            # just word itself 
            if word in word_dict:
                feature_matrix[line_count][word_dict[word]] += 1

            try: 
                # lower capiticalization of the word
                lower_word = str(word).lower()
                if lower_word in word_dict:
                    feature_matrix[line_count][word_dict[lower_word]] += 1
                # no punctuation 
                replace_punctuation = str(word).maketrans(string.punctuation, '')
                clean_word = str(word).translate(replace_punctuation)

                if clean_word in word_dict:
                    feature_matrix[line_count][word_dict[clean_word]] += 1
            except:
                continue

        line_count += 1

    print("done with feature_matrix")

    return feature_matrix

# ----------------------------------------------------------
def main():
    #try:
    #    dy.cg_revert()
    #except:
    dy.renew_cg()

    train_file = open("%s" %(sys.argv[1]))
    test_file = open("%s" %(sys.argv[2]))
    train_text_set, train_content_label_set, train_type_label_set, unique_content, unique_type = extract_from_json(train_file)
    test_text_set, test_content_label_set, test_type_label_set, _, _ = extract_from_json(test_file)
    

    word_dict = {}
    word_dict = extract_dictionary(train_text_set, word_dict)
    word_dict = extract_dictionary(test_text_set, word_dict)

    train_feature_matrix = generate_feature_matrix(train_text_set, word_dict)
    test_feature_matrix = generate_feature_matrix(test_text_set, word_dict)


    features_total = len(train_feature_matrix[0])
    para_collec = dy.ParameterCollection()
    pW1 = para_collec.add_parameters((150, 200), dy.NormalInitializer())
    pBias1 = para_collec.add_parameters((150), dy.ConstInitializer(0))
    pW2_content = para_collec.add_parameters((100, 150), dy.NormalInitializer())
    pBias2_content = para_collec.add_parameters((100), dy.ConstInitializer(0))
    pW3_content = para_collec.add_parameters((len(unique_content), 100), dy.NormalInitializer())
    pBias3_content = para_collec.add_parameters((len(unique_content)), dy.ConstInitializer(0))
    pW2_type = para_collec.add_parameters((50, 150), dy.NormalInitializer())
    pBias2_type = para_collec.add_parameters((50), dy.ConstInitializer(0))
    pW3_type = para_collec.add_parameters((len(unique_type), 50), dy.NormalInitializer())
    pBias3_type = para_collec.add_parameters((len(unique_type)), dy.ConstInitializer(0))
    lookup = para_collec.add_lookup_parameters((features_total, 200), dy.NormalInitializer())

    trainer = dy.SimpleSGDTrainer(para_collec)
    #print(X_train)
    
    for i in range(0, 1):
        # resample minority and majority classes
        majority, majority_content_label, majority_type_label, minority, minority_content_label, minority_type_label = label_separator("type", train_feature_matrix, train_content_label_set, train_type_label_set)
        minority_u_text, minority_u_content_label, minority_u_type_label = resample(minority, minority_content_label, minority_type_label, replace=True, n_samples=int(len(majority) * 3), random_state=123)

        #X_train = majority + minority_u_text
        #y_train_content = majority_content_label + minority_u_content_label
        #y_train_type = majority_type_label + minority_u_type_label
        
        X_train = train_feature_matrix
        y_train_content = train_content_label_set
        y_train_type = train_type_label_set

        #for index in range(0, len(X_train)):
        for index in range(0, 500):

            w1 = dy.parameter(pW1)
            bias1 = dy.parameter(pBias1)
            w2_content = dy.parameter(pW2_content)
            bias2_content = dy.parameter(pBias2_content)
            w3_content = dy.parameter(pW3_content)
            bias3_content = dy.parameter(pBias3_content)
            w2_type = dy.parameter(pW2_type)
            bias2_type = dy.parameter(pBias2_type)
            w3_type = dy.parameter(pW3_type)
            bias3_type = dy.parameter(pBias3_type)
            
            input_text = []
            #line = train_text_set[index]
            input_array = X_train[index]
            #print(X_train[index].size)
            for i in range(0, X_train[index].size):
                if X_train[index][i] > 0:
                    input_text.append(lookup[X_train[index][i]])

            x = dy.concatenate(input_text, 1)
            #print(x.npvalue().shape)
            #print(dy.sum_dim(x, [1]).npvalue().shape)
            e_in = dy.sum_dim(x, [1])/features_total
            e_affin1 = dy.affine_transform([bias1, w1, e_in])
            e_affin1 = dy.rectify(e_affin1)
            e_content_affin2 = dy.affine_transform([bias2_content, w2_content, e_affin1])
            e_content_affin2 = dy.dropout(e_content_affin2, 0.5)
            e_content_affin2 = dy.rectify(e_content_affin2)
            e_content_affin3 = dy.affine_transform([bias3_content, w3_content, e_content_affin2])
            e_content_affin3 = dy.dropout(e_content_affin3, 0.5)
            e_content_affin3 = dy.rectify(e_content_affin3)
            e_type_affin2 = dy.affine_transform([bias2_type, w2_type, e_affin1])
            e_type_affin2 = dy.dropout(e_type_affin2, 0.5)
            e_type_affin2 = dy.rectify(e_type_affin2)
            e_type_affin3 = dy.affine_transform([bias3_type, w3_type, e_type_affin2])
            e_type_affin3 = dy.dropout(e_type_affin3, 0.5)
            e_type_affin3 = dy.rectify(e_type_affin3)
            content_output = dy.pickneglogsoftmax(e_content_affin3, y_train_content[index])
            content_loss = content_output.scalar_value()
            type_output = dy.pickneglogsoftmax(e_type_affin3, y_train_type[index])
            type_loss = type_output.scalar_value()
            
            if index % 100 == 0:
                print(index, ": content_loss: ", content_loss, "type_loss", type_loss)
            
            content_output.backward()
            trainer.update()
            type_output.backward()
            trainer.update()

            dy.cg_checkpoint()

    print("testing...")
    pred_content = []
    pred_type = []

    w1 = dy.parameter(pW1)
    bias1 = dy.parameter(pBias1)
    w2_content = dy.parameter(pW2_content)
    bias2_content = dy.parameter(pBias2_content)
    w3_content = dy.parameter(pW3_content)
    bias3_content = dy.parameter(pBias3_content)
    w2_type = dy.parameter(pW2_type)
    bias2_type = dy.parameter(pBias2_type)
    w3_type = dy.parameter(pW3_type)
    bias3_type = dy.parameter(pBias3_type)

    for index in range(0, len(test_feature_matrix)):
        # x = dy.inputTensor(test_feature_matrix[index])
       
        input_text = []
        line = train_text_set[index]
        for word in line:
            # check if RT
            if word == "RT":
                input_text.append(lookup[len(word_dict)])
            # check if hashtag
            if word[0] == "#":
                input_text.append(lookup[len(word_dict) + 1])

            # check if mention
            if word[0] == "@":
                input_text.append(lookup[len(word_dict) + 2])

            # just word itself 
            if word in word_dict:
                input_text.append(lookup[word_dict[word]])

            try: 
                # lower capiticalization of the word
                lower_word = str(word).lower()
                input_text.append(lookup[word_dict[lower_word]])
                # no punctuation 
                replace_punctuation = str(word).maketrans(string.punctuation, '')
                clean_word = str(word).translate(replace_punctuation)
                input_text.append(lookup[word_dict[clean_word]])
            except:
                continue

        e_in = dy.sum_dim(x, [1])/features_total
        e_affin1 = dy.affine_transform([bias1, w1, e_in])
        e_affin1 = dy.rectify(e_affin1)
        e_content_affin2 = dy.affine_transform([bias2_content, w2_content, e_affin1])
        e_content_affin2 = dy.rectify(e_content_affin2)
        e_content_affin3 = dy.affine_transform([bias3_content, w3_content, e_content_affin2])
        e_content_affin3 = dy.rectify(e_content_affin3)
        e_type_affin2 = dy.affine_transform([bias2_type, w2_type, e_affin1])
        e_type_affin2 = dy.rectify(e_type_affin2)
        e_type_affin3 = dy.affine_transform([bias3_type, w3_type, e_type_affin2])
        e_type_affin3 = dy.rectify(e_type_affin3)
        content_output = np.argmax(e_content_affin3.npvalue())
        pred_content.append(content_output)
        type_output = np.argmax(e_type_affin3.npvalue())
        pred_type.append(type_output)

    misclassification_content = 0
    misclassification_type = 0
    for index in range(0, len(pred_content)):
        #print(pred_content[index], test_content_label_set[index])
        if pred_content[index] != test_content_label_set[index]:
            misclassification_content += 1
        if pred_type[index] != test_type_label_set[index]:
            misclassification_type += 1
    
    print("content acc: ", (1 - float(misclassification_content/len(pred_content))))
    print("type acc: ", (1 - float(misclassification_type/len(pred_type))))

# ----------------------------------------------------------

if __name__ == '__main__':
    main()
