# Notes

I think I will start with a markov chain (at present an n-gram based markov chain), though I have ideas to use a neural network and word vector representations in the future.

A few ideas:

* I think it would be intersting to show poems in the style of this or that, which could just replace words with similar words according to their cosine distance to other words embedded into a different vector space. Say you wanted to see what dickenson's poem would look like if written by a man: use a man's vector space embedding and pick the most common similar words that still maximize the probability of the poem according to the viterbi algorithm. 


## TODO
* implememnt a markov chain algorithm.
* implement viterbi
* get gensim synonyms working
* possibly get another synonym (wordnet, etc.) algorithm working

