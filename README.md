*To "live at ease" is There; and, to these divine beings, verity is mother and nurse, existence and sustenance; all that is not of process but of authentic being they see, and themselves in all: for all is transparent, nothing dark, nothing resistant; every being is lucid to every other, in breadth and depth; light runs through light. And each of them contains all within itself, and at the same time sees all in every other, so that everywhere there is all, and all is all and each all, and infinite the glory.*

-The Enneads of Plotinus: The Fifth Ennead, Eighth Tractate

# All is all - Poetry Generation Engine

This is a python project that generates poetry with neural networks. ...Or it will be.

## Setup


1.) Install dependencies: 
* [nltk](https://www.nltk.org/install.html)
* [gensim](https://radimrehurek.com/gensim/install.html)
    * `easy_install -U gensim`
* [Keras w/ tensorflow backend](https://keras.io/#installation)

2.) Download the CMU pronunciation dictionary (requires git):

	cd corpus
	./get_cmu_dict.sh

3.) Run tests:

     cd src
     python corpus_test.py

# To do

Finish reading 'Beyond Narrative Description: Generating Poetry from Images by Multi-Adversarial Training'

Find the corpus used for that paper ^. Start thinking about other ways to create a corpus (can we use topic modeling or just vector similarity). Basically we want to have some sort of corous of any type of text and poems corresponding to each text. What are some good ideas?? What domain has obvious poem matches? Maybe we don't even want to do anything that compllicated at first... like let's start with just a few very simple phrases and be able to generate poems from them. News articles might be fun. A better way to do senence simplification.


Create more corpus entry points. Be able to generate a corpus from:
* Tweets (using tweepy)
* CONLL Formatted Data
* Sentences seperated by new lines. (Done)

# The Information

Here are a few different surveys and introductions to the field of generative poetry that I've found useful.

*Pentametron*

https://twitter.com/pentametron?lang=en

*Andrej Karpathy*

https://github.com/karpathy/char-rn

*Journal of Mathematics and the Arts - A taxonomy of generative poetry techniques*

https://www.tandfonline.com/doi/abs/10.1080/17513472.2017.1373561?journalCode=tmaa20

http://archive.bridgesmathart.org/2016/bridges2016-195.pdf

*When you Reach Kyoto - Brian Kim Stefans*

http://collection.eliterature.org/1/works/geniwate__generative_poetry.html

*From Dust to Deep Learning: an Intro to Generative Poetry*

http://interaccess.org/workshop/2017/aug/dust-deep-learning-intro-generative-poetry

*Beyond Narrative Description: Generating Poetry from Images by Multi-Adversarial Training*

https://arxiv.org/abs/1804.08473

* Generating Text with Recurrant Neural Networks *

https://www.cs.utoronto.ca/~ilya/pubs/2011/LANG-RNN.pdf


# Coding Tutorials

* Text Generation With LSTM Recurrent Neural Networks in Python with Keras *

https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/


# Other

*Quote from the Plotinus, THE SIX ENNEADS*

http://www.sacred-texts.com/cla/plotenn/enn481.htm

*Gears ASCII art from* 

https://groups.google.com/forum/#!topic/alt.ascii-art/kgiB6eu0Gy0

                                _   _
                               ( \_/ )
                  _   _       __) _ (__
          _   _  ( \_/ )  _  (__ (_) __)
         ( \_/ )__) _ (__( \_/ )) _ (
        __) _ ((__ (_) __)) _ ((_/ \_)
       (__ (_) __)) _ ((__ (_) __)
          ) _ (  (_/ \_)  ) _ (
         (_/ \_)         (_/ \_)
         
