P# All is all - Poetry Generation Engine

This is a python project that generates poetry with neural networks. ...Or it will be.

## Setup

1.) Download the CMU pronunciation dictionary:

	cd corpus
	./get_cmu_dict.sh
	
2.) Install dependencies: 

* Gensim
* Keras w/ tensorflow backend
* nltk

# To Do

Create more corpus entry points. Be able to generate a corpus from:
* Tweets (using tweepy)
* CONLL Formatted Data
* Sentences seperated by new lines. (Done)
