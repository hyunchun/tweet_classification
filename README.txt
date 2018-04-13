OP 2017-2018 Artificial Intelligence for Language / Computational Linguistics: Tweet Classification

Below is the result of UROP program I worked on during 2017-2018.  
> // link to mentor's and my page (for in future) 
> // poster (image || pdf)
name: Hyun A Chung

1) data: contains both labeled and raw version of tweet datasets
	Datasets are imported from (1) CMU Tweet NLP and (2) Broad Corpus dataset
	(1) CMU Tweet NLP: 
	      	* "Oct27": 1827 tweets from 2010-10-27
      		* "Daily547": 547 tweets, one per day from 2011-01-01 through 2012-06-30
		
		link to website: http://www.cs.cmu.edu/~ark/TweetNLP/
		link to github: https://github.com/brendano/ark-tweet-nlp
		
  References:

    The following papers describe this dataset.  If you use this data in a
    research publication, we ask that you cite this (the original paper):

    Kevin Gimpel, Nathan Schneider, Brendan O'Connor, Dipanjan Das,
      Daniel Mills, Jacob Eisenstein, Michael Heilman, Dani Yogatama, Jeffrey
      Flanigan, and Noah A. Smith.

    Part-of-Speech Tagging for Twitter: Annotation, Features, and Experiments.
    In Proceedings of the Annual Meeting of the Association for Computational
      Linguistics, companion volume, Portland, OR, June 2011.

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
		// include installation list 
		// complete example of command
			command: python classification_dynet.py <train_file> <test_file>
		b) classification_scikit-learn.py: classifies tweets according to type label and content label using scikit-learn library
			command: python classification_scikit-learn.py <train_file> <test_file>
		
		// explanation of labels
		// add explanation of other python files (move those files to script)
		// include tweet download script (no API key)
		// license for both data and code (separate: code -> whatever I want to, data -> follow license rules on each github)
		// add description to top of repository
		// data analysis


