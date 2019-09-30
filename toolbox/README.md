## Setup

1.) Install dependencies: 
* [nltk](https://www.nltk.org/install.html)
* [gensim](https://radimrehurek.com/gensim/install.html)
    * `easy_install -U gensim`
* [Keras w/ tensorflow backend](https://keras.io/#installation)
* [pdftotext](https://www.xpdfreader.com/pdftotext-man.html) (for importing gandhi corpus).

2.) Download the CMU pronunciation dictionary (requires git):

	cd toolbox/corpus
	./get_cmu_dict.sh

3.) Run tests:

     cd toolbox/src
     python corpus_test.py
