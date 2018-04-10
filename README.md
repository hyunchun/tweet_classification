#UROP Tweet Classification Project
##name: Hyun A Chung

last updated: 2018-04-10

##folder:_
	1) data: contains both labeled and raw version of tweet datasets_
	Datasets are imported from (1) CMU Tweet NLP and (2) Broad Corpus dataset_
	(1) CMU Tweet NLP: 
		link to website: http://www.cs.cmu.edu/~ark/TweetNLP/
		link to github: https://github.com/brendano/ark-tweet-nlp
	(2) Broad Twitter Corpus:
		link to paper: http://aclweb.org/anthology/C16-1111 
		link to github: https://github.com/GateNLP/broad_twitter_corpus
	
	'data' folder contains two folders: 
		a) labeled: contains labeled version of tweet datasets
			i) CMU_1827_labeled: labeled version of CMU_1827 dataset in form of custom json format
			ii) CMU_547_labeled: labeled version of CMU_547 dataset in form of custom json format
			iii) h_labeled: labeled version of h dataset in form of custom json format
		b) raw: contains raw version of tweet datasets
			i) CMU_547_raw_tsv: correspond to daily547.tweets.json.tsv 
			ii) CMU_1827_raw_tsv: correspond to oct27.tweets.json.tsv
			iii) a_json, b_json, e_json, f_json, g_json, h_json: no label .json version of Broad Twitter Corpus datasets 
	2) classifiers: contains classification_dynet.py and classification_scikit-learn.py
		a) classification_dynet.py: classifies tweets according to type label and content label using dynet library
			form: python classification_dynet.py <train_file> <test_file>
		b) classification_scikit-learn.py: classifies tweets according to type label and content label using scikit-learn library

