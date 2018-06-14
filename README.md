*To "live at ease" is There; and, to these divine beings, verity is mother and nurse, existence and sustenance; all that is not of process but of authentic being they see, and themselves in all: for all is transparent, nothing dark, nothing resistant; every being is lucid to every other, in breadth and depth; light runs through light. And each of them contains all within itself, and at the same time sees all in every other, so that everywhere there is all, and all is all and each all, and infinite the glory.*

-The Enneads of Plotinus: The Fifth Ennead, Eighth Tractate

# All is all - Poetry Generation Engine

This is a python project that generates poetry with neural networks. ...Or it will be.

## Setup


1.) Install dependencies: 
* nltk
* [gensim](https://radimrehurek.com/gensim/install.html)

    easy_install -U gensim
    
    or
    
    pip install --upgrade gensim
    
* Keras w/ tensorflow backend

2.) Download the CMU pronunciation dictionary:

	cd corpus
	./get_cmu_dict.sh

# To Do

Create more corpus entry points. Be able to generate a corpus from:
* Tweets (using tweepy)
* CONLL Formatted Data
* Sentences seperated by new lines. (Done)
