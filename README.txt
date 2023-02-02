Mike Rabayda
What are words that, specifically for the collection in data/obits.json, appear in a lot of documents and are thus not helpful query terms?

The top 5 words with highest tf-idf score that appear in a lot of documents are Khrushchev, Soviet, Stalin, party, and Ukraine.

Modify the main method in tfidf.py so that you convert Karen Spark Jones’ obituary into a document vector based on the vocabularies developed in training. 
Based on tf-idf, what words are indicative of her obituary? Why might you think these words are indicative of Karen Spark Jones’s obituary.

Jones, computer, Cambridge, computers, and she were the top 5 most indicative words of her obituary.  These words make sense to be the most indicative of 
her obituary, as you would expect that the obituary talks a lot about Jones herself and her accomplishments in computer science at Cambridge since she worked in a research
lab at Cambridge University when researching NLP.  Since obituaries are typically summaries of the great parts of the deceased's life, we would expect there to be a lot of
mentions of her research using computers where she worked, and the results confirm that.
