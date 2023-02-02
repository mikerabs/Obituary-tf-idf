import argparse
import json
from collections import defaultdict
from math import log
import os

from typing import Iterable, Tuple, Dict

from nltk.tokenize import TreebankWordTokenizer
from nltk import FreqDist

kUNK = "<UNK>"

def log10(x):
    return log(x) / log(10)

def lower(str):
    return str.lower()


class TfIdf:
    """Class that builds a vocabulary and then computes tf-idf scores
    given a corpus.

    """

    def __init__(self, vocab_size=10000,
                 tokenize_function=TreebankWordTokenizer().tokenize,
                 normalize_function=lower, unk_cutoff=2):
        self._vocab_size = vocab_size
        self._total_docs = 0

        self._vocab_final = False
        self._vocab = {}
        self._unk_cutoff = unk_cutoff

        self._tokenizer = tokenize_function
        self._normalizer = normalize_function

        # Add your code here!

        self._countWords = {}#keeps track of all words in all documents
        self._wordsPerDoc = {}#keeps track of all words in every document
        self._docs = []



    def train_seen(self, word: str, count: int=1):
        """Tells the language model that a word has been seen @count times.  This
        will be used to build the final vocabulary.

        word -- The string represenation of the word.  After we
        finalize the vocabulary, we'll be able to create more
        efficient integer representations, but we can't do that just
        yet.

        count -- How many times we've seen this word (by default, this is one).
        """

        assert not self._vocab_final, \
            "Trying to add new words to finalized vocab"

        # Add your code here!

        #check to see if word is alreadu in CW dict, update number
        countInCW = 0
        numDocsIn = 0
        if word in self._countWords:
            countInCW += self._countWords[word][0]
            numDocsIn += self._countWords[word][1]

        
        #adding one or whatever the count is set to in the argument
        countInCW += count
        #adding one to the numDocs a given word is in
        numDocsIn += 1

        #set the # of docs value equal to 1 since we haven't seen this word
        self._countWords[word] = (countInCW,numDocsIn)


        

        

        
        




    def add_document(self, text: str):
        """
        Tokenize a piece of text and add the entries to the class's counts.

        text -- The raw string containing a document
        """
        self._wordsPerDoc = {}#reset wordsPerDoc since it is new doc
        self._total_docs +=1
        

        for word in self.tokenize(text):#tokenize is returning the hashed vocab ID
            #need the string form of the word, if word in vocab then it returns its ID
            
            #adjust for the case in which tokenize returns the string c
            countWPD = 0
            if word in self._wordsPerDoc:
                countWPD = self._wordsPerDoc[word]

            countWPD += 1
            self._wordsPerDoc[word] = countWPD

            #self.train_seen(word)#don't use train seen, vocab is already finalized
        self._docs.append(self._wordsPerDoc)

            

    def tokenize(self, sent: str) -> Iterable[int]:
        """Return a generator over tokens in the sentence; return the vocab
        of a sentence if finalized, otherwise just return the raw string.

        sent -- A string

        """

        # You don't need to modify this code.
        for ii in self._tokenizer(sent):
            if self._vocab_final:
                yield self.vocab_lookup(ii)
            else:
                yield ii

    def doc_tfidf(self, doc: str) -> Dict[Tuple[str, int], float]:
        """Given a document, create a dictionary representation of its tfidf vector

        doc -- raw string of the document"""

        counts = FreqDist(self.tokenize(doc))
        d = {}
        for ii in self._tokenizer(doc):
            ww = self.vocab_lookup(ii)
            d[(ww, ii)] = counts.freq(ww) * self.inv_docfreq(ww)
        return d
                
    def term_freq(self, word: int) -> float:
        """Return the frequence of a word if it's in the vocabulary, zero otherwise.

        word -- The integer lookup of the word.
        """
        #FIX THIS

        if word in self._vocab.values() and len(self._docs) == 1:
            #RETURN THE FREQUENCY NOT THE COUNT
             #tf = | times w appears in doc|/|num tokens in doc|
            timesW = abs(self._docs[0][word])
            numTokens = 0
            for key in self._docs[0].keys():
                numTokens += self._docs[0][key]
            #numTokens = abs(len(self._docs[0].keys()))

            return float(timesW/numTokens)
        
        return 0.0


    def inv_docfreq(self, word: int) -> float:
        """Compute the inverse document frequency of a word.  Return 0.0 if
        the word has never been seen.

        Keyword arguments:
        word -- The word to look up the document frequency of a word.

        """

        #need number of documents that contain w, code into CW?

        #idf = numDocuments/number of documents that contain w
        if word in self._vocab.values():
            numDocuments = abs(self._total_docs)
            numDocsContainingW = 0
            for i in range(len(self._docs)):
                #on second time around, word code is not in docs[1], must handle
                if word not in self._docs[i].keys():
                    continue
                if float(abs(self._docs[i][word])/abs(len(self._docs[i].keys()))) != 0:
                    numDocsContainingW += 1

            return log10(numDocuments/numDocsContainingW)#can't refer to countWords using hashed value
        #self._countWords[word][1]
        return 0.0

    def vocab_lookup(self, word: str) -> int:
        """
        Given a word, provides a vocabulary integer representation.  Words under the
        cutoff threshold shold have the same value.  All words with counts
        greater than or equal to the cutoff should be unique and consistent.

        This is useful for turning words into features in later homeworks.
        In HW01 we did not specify how to represent each word, here we are using integers

        word -- The word to lookup
        """
        assert self._vocab_final, \
            "Vocab must be finalized before looking up words"

        if word in self._vocab:
            return self._vocab[word]
        else:
            return self._vocab[kUNK]# this is where we are getting tripped up

    def finalize(self):
        """
        Fixes the vocabulary as static, prevents keeping additional vocab from
        being added
        """

        # Add code to generate the vocabulary that the vocab lookup
        # function can use!


        for word in self._countWords.keys():
            if self._countWords[word][0] >= self._unk_cutoff:
                self._vocab[word] = hash(word)
                #instructions say to map word to ID number, does this imply to not use count as value?
                #maybe hash the string?vself._countWords[word][0]
            else:
                self._vocab[kUNK] = hash(kUNK)

        if kUNK not in self._vocab.keys():
            self._vocab[kUNK] = hash(kUNK)
        self._vocab_final = True

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--root_dir", help="Obituatires",
                           type=str, default='data/',
                           required=False)
    argparser.add_argument("--train_dataset", help="Dataset for training",
                           type=str, default='obits.json',#used to be obits.train.json
                           required=False)
    argparser.add_argument("--test_dataset", help="Dataset for test",
                           type=str, default='sparck-jones.txt',
                           required=False)
    argparser.add_argument("--limit", help="Number of training documents",
                           type=int, default=-1, required=False)
    args = argparser.parse_args()

    vocab = TfIdf()

    with open(os.path.join(args.root_dir, args.train_dataset)) as infile:
        data = json.load(infile)["obit"]
        if args.limit > 0:
            data = data[:args.limit]
        for ii in data:
            for word in vocab.tokenize(data[ii]):
                vocab.train_seen(word)
        vocab.finalize()

        for ii in data:
            vocab.add_document(data[ii])

    with open(os.path.join(args.root_dir, args.test_dataset), encoding = 'utf-8') as infile:#train_dataset
        data = infile.read()#json.load(infile)["obit"]
        vector = vocab.doc_tfidf(data)#['0'])
        for word, tfidf in sorted(vector.items(), key=lambda kv: kv[1], reverse=True)[:50]:
            print("%s:%i\t%f" % (word[1], word[0], tfidf))