# Notes

A few ideas:

* I think it would be intersting to show poems in the style of this or that, which could just replace words with similar words according to their cosine distance to other words embedded into a different vector space. Say you wanted to see what dickenson's poem would look like if written by a man: use a man's vector space embedding and pick the most common similar words that still maximize the probability of the poem according to the viterbi algorithm. 


## TODO
* implement viterbi
* get gensim synonyms working
* possibly get another synonym (wordnet, etc.) algorithm working