============================================================================

UROP 2017-2018 Artificial Intelligence for Language / Computational Linguistics: Tweet Classification
Hyun A Chung, Undergraduate Student
Jonathan Kummerfeld, Post-doctoral Research Fellow
	http://jkk.name/

============================================================================

Description:

This repository contains the result of Tweet Classification project for UROP 
that I worked on during 2017-2018.
	
Abstract:

Social media is widely used as a communication tool by billions of people. 
For most of the existing social media, the feeds are ordered in reverse-
chronological order. To give flexibility to a user’s experience, especially
for Twitter, this project seeks to create a classification model that can 
categorize a tweet (a short online message by a Twitter user) to customize 
a user’s social media feed. This project involves two parts: data collection
and model construction. For the data collection, we collected existing tweet
datasets and designed labeling schemes for the tweets based on their type 
and content. For the model construction, we developed type-based and content
-based categorization models using DyNet and Scikit-learn machine learning 
Python libraries. As the result, the DyNet model gave 46.6% accuracy for the 
content labels and 66.3% accuracy for the type labels. Scikit-learn model 
gave 45.2% accuracy for the content labels and 65.9% accuracy for type labels.
Using the built models, our overall goal is to create an application that 
receives user’s tweets in real-time from the Twitter feed, then performs 
live classification to produce customized feed only with the selected labels 
or a user’s customized labels.  

Data: 

Datasets are imported from (1) CMU Tweet NLP and (2) Broad Corpus dataset
(1) CMU Tweet NLP: 
	* "Oct27": 1827 tweets from 2010-10-27
	* "Daily547": 547 tweets, one per day from 2011-01-01 through 2012-06-30
			
	http://www.cs.cmu.edu/~ark/TweetNLP/
	https://github.com/brendano/ark-tweet-nlp

	The following papers describe this dataset.  If you use this data in a
	research publication, we ask that you cite this (the original paper):

	Kevin Gimpel, Nathan Schneider, Brendan O'Connor, Dipanjan Das,
		Daniel Mills, Jacob Eisenstein, Michael Heilman, Dani Yogatama, Jeffrey
		Flanigan, and Noah A. Smith.

	Part-of-Speech Tagging for Twitter: Annotation, Features, and Experiments.
		Proceedings of the Annual Meeting of the Association for Computational
		Linguistics, companion volume, Portland, OR, June 2011.

(2) Broad Twitter Corpus:
	* "H":	General collection of 2000 non-UK tweets from 2014
			
	http://aclweb.org/anthology/C16-1111 
	https://github.com/GateNLP/broad_twitter_corpus
		
	The following papers describe this dataset.  If you use this data in a
	research publication, we ask that you cite this (the original paper):
			
	Broad Twitter Corpus: A Diverse Named Entity Recognition Resource. Leon Derczynski, 
		Kalina Bontcheva, and Ian Roberts. Proceedings of COLING, pages 1169-1179 2016.
				
	The paper's full open access, and can be found easily; here's one link: http://www.aclweb.org/anthology/C16-1111    
	
'data' folder contains two folders: 
a) labeled: contains labeled version of tweet datasets
	i) CMU_1827_labeled: labeled version of CMU_1827 dataset in form of custom json format
	ii) CMU_547_labeled: labeled version of CMU_547 dataset in form of custom json format
	iii) h_labeled: labeled version of h dataset in form of custom json format
b) raw: contains raw version of tweet datasets
	i) CMU_547_raw_tsv: correspond to daily547.tweets.json.tsv 
	ii) CMU_1827_raw_tsv: correspond to oct27.tweets.json.tsv
	iii) a_json, b_json, e_json, f_json, g_json, h_json: no label .json version of Broad Twitter Corpus datasets 

Classifiers:

In this project, two different classifiers were implemented: classification_dynet.py and classification_scikit-learn.py
a) classification_dynet.py
	Requires dynet and numpy libraries
	dynet: http://dynet.readthedocs.io/en/latest/index.html
	numpy: http://www.numpy.org/
	
	python classification_dynet.py <train_file> <test_file>
	e.g. python classification_dynet.py train_set test_set
	<train_file> and <test_file>: json files with custom json format for this project
	
b) classification_scikit-learn.py
	Requires sklearn and numpy libraries
	sklearn: http://scikit-learn.org/stable/
	numpy: http://www.numpy.org/

	python classification_scikit-learn.py <train_file> <test_file>
	e.g. python classification_scikit-learn.py train_set test_set
	<train_file> and <test_file>: json files with custom json format for this project
		
		// explanation of labels
		// add explanation of other python files (move those files to script)
		// include tweet download script (no API key)
		// license for both data and code (separate: code -> whatever I want to, data -> follow license rules on each github)
		// add description to top of repository
		// data analysis


