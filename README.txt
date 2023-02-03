Mike Rabayda


Writeup:
What are words that, specifically for the collection in data/obits.json, appear in a lot of documents and are thus not helpful query terms?

The top 5 words with highest tf-idf score that appear in a lot of documents are Khrushchev, Soviet, Stalin, party, and Ukraine.

Modify the main method in tfidf.py so that you convert Karen Spark Jones’ obituary into a document vector based on the vocabularies developed in training. 
Based on tf-idf, what words are indicative of her obituary? Why might you think these words are indicative of Karen Spark Jones’s obituary.

Jones, computer, Cambridge, computers, and she were the top 5 most indicative words of her obituary.  These words make sense to be the most indicative of 
her obituary, as you would expect that the obituary talks a lot about Jones herself and her accomplishments in computer science at Cambridge since she worked in a research
lab at Cambridge University when researching NLP.  Since obituaries are typically summaries of the great parts of the deceased's life, we would expect there to be a lot of
mentions of her research using computers where she worked, and the results confirm that.




TF-iDF:
Why do you think it might be better to use sklearn (or another libraries) implementation compared to your own?

It ensures that the implementation of whatever process we are trying to implement is sure to work.  We can make mistakes using the library's implementation
but we are always guaranteed that the implementation will perform its function properly if used properly.  

Make sure you understand what is happening in each cell. Starting at the 4th cell (the one with the from sklearn.feature_extraction.text 
import TfidfVectorizer line), write a brief comment explaning what is happening in each cell.

4: This cell imports the TfidfVectorizer object from sklearn and creates an instance of that object right after called vectorizer.  
It then sets a variable called transformed_documents equal to output of our new instance's method fit_transform() setting all_docs as the argument.

5: This cell sets a variable equal to the array form of transformed_documents, and then prints the length of said array, or the number of documents.

6: This cell imports pandas, and then creates an output folder.

7: This cell creates a list of output filepaths for all textfiles.  It then loops through each item in the array form of transformed_documents, and then
creates a dataframe for each document using the TfidfVectorizer instance, and outputs this dataframe to a csv files in the output_filenames array. 

Choose any obituary and compare the document representation in that notebook with the document representation for the same obituary from your 
tf-idf.py implementation. What differences do you see?

When comparing our implementation of tf-idf to sklearn's implementation looking at the Khrushchev obituary, we find that most of the same word contained within our top 5 also 
scored within the top 5 in the sklearn implementation as Khrushchev, soviet, and Stalin are similarly the top 3.  The main diffference between both implementations is the magnitude
of the tf-idf score, as all of the scores in the sklearn's implementation that are not equal to zero are above 1, ranging from 1.4 to 123 which far outweighs our range of 0.0007 to 0.016.

What changes do you think you’d have to make to the settings of the TfidVectorizer object in the notebook to make the results be more similar 
to the results in your implementation.

I imagine I'd have to adjust the max_df and min_df settings within the arguments while instantiating the TfidVectorizer object vectorizer and adjust them to so that the scores would be
more similar to the results in my implementation.


Feedback:
How long did you spend on the assignment?

I spent about 10 hours on this project.

What did you learn by doing this assignment?

Learned how to implement tf-idf myself and how my own implementation differs from a formal library's implementation.

Briefly describe a Computational Text Analysis/Text as Data research question where using tf-idf 
would be useful (keep this answer to a maximum of 3 sentences).

What are the most talked about issues when congressmen debate specific policy changes?  We could read a corpus composed of transcripts 
from those congressional sessions and determine  which terms are strongly associated with debating upon a certain policy.
