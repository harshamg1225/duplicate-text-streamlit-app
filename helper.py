import re
import nltk
import pandas as pdclr
import numpy as np
import pickle
from bs4 import BeautifulSoup
import distance
from fuzzywuzzy import fuzz
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("punkt_tab")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess


word_vec=pickle.load(open("word_vec.pkl","rb"))

def test_common_words(q1,q2):
    q1=q1.split()
    q2=q2.split()
    return len(set(q1).intersection(set(q2)))



def test_total_words(q1,q2):
    return len(q1.split())+len(q2.split())

## token features
def test_fetech_token_features(q1,q2):

  SAFE_DIV=0.0001
  token_features=[0.0]*8
  stopword=stopwords.words("english")

  

  # conversting the sentence into token
  q1_tokens=q1.split()
  q2_tokens=q2.split()

  if len(q1_tokens)==0 or len(q2_tokens)==0:
    return token_features

  # Get the common word count

  q1_words=[word for word in q1_tokens if word not in stopword]
  q2_words=[word for word in q2_tokens if word not in stopword]

  q1_words=set(map(lambda word:word.lower().strip(),q1_words))
  q2_words=set(map(lambda word:word.lower().strip(),q2_words))

  common_word_count=len(q1_words.intersection(q2_words))

  token_features[0]=common_word_count/(min(len(q1_words),len(q2_words))+SAFE_DIV)
  token_features[1]=common_word_count/(max(len(q1_words),len(q2_words))+SAFE_DIV)

  # get the common stopword

  q1_stopwords=[word for word in q1_tokens if word in stopword]
  q2_stopwords=[word for word in q2_tokens if word in stopword]

  q1_stopwords=set(map(lambda word:word.lower().strip(),q1_stopwords))
  q2_stopwords=set(map(lambda word:word.lower().strip(),q2_stopwords))

  common_stop_word_count=len(q1_stopwords.intersection(q2_stopwords))
  token_features[2]=common_stop_word_count/(min(len(q1_stopwords),len(q2_stopwords))+SAFE_DIV)
  token_features[3]=common_stop_word_count/(max(len(q1_stopwords),len(q2_stopwords))+SAFE_DIV)

  # common token count

  q1_token=set(q1)
  q2_token=set(q2)

  common_token_count=len(q1_token.intersection(q2_token))
  token_features[4]=common_token_count/(min(len(q1_token),len(q2_token))+SAFE_DIV)
  token_features[5]=common_token_count/(max(len(q1_token),len(q2_token))+SAFE_DIV)

  # Last word of both question is same or not
  token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])

  # first word of both question is same or not
  token_features[7] = int(q1_tokens[0] == q2_tokens[0])

  return token_features

# length based features

def test_fetech_length_features(q1,q2):

  length_features=[0.0]*3



  len_q1=len(q1.split())
  len_q2=len(q2.split())

  if q1==0 or q2==0:
    return length_features

  # mean length
  length_features[0]=(len_q1+len_q2)/2

  # absolute diffrence

  length_features[1]=abs(len_q1-len_q2)


  # longest subtring ratio

  strg=list(distance.lcsubstrings(q1,q2))

  length_features[2]=len(strg[0])/(min(len(q1),len(q2)+1))

  return length_features



def test_fetch_fuzzy_features(q1,q2):

  fuzzy_features=[0.0]*4

  fuzzy_features[0]=fuzz.QRatio(q1,q2)
  fuzzy_features[1]=fuzz.partial_ratio(q1,q2)
  fuzzy_features[2]=fuzz.token_sort_ratio(q1,q2)
  fuzzy_features[3]=fuzz.token_set_ratio(q1,q2)

  return fuzzy_features


def Preprocess(q):

  q=str(q).lower().strip()

  # replace certain special Charcters with their string equivalents

  q=q.replace("%"," percent ")
  q=q.replace("₹"," rupee ")
  q=q.replace("$"," dollar ")
  q=q.replace("€"," euro ")
  q=q.replace("@"," at ")

  # The pattern '[math]' appears around 900 time in the whole dataset

  q=q.replace("[math]","")

  # replace some number with string equivalents

  q=q.replace(",000,000,000 ","b")
  q=q.replace(",000,000 ","m")
  q=q.replace(",000 ","k")

  q=re.sub(r"([0-9]+)000000000",r"\1b",q)
  q=re.sub(r"([0-9]+)000000",r"\1m",q)
  q=re.sub(r"([0-9]+)000",r"\1k",q)

  # Deconstructing words
  # https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions
  # https://stackoverflow.com/a/19794953

  contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    }

  q=" ".join([contractions[word] if word in contractions else word for word in q.split()])

  q=q.replace("'ve"," have ")
  q=q.replace("n't"," not ")
  q=q.replace("'re"," are ")
  q=q.replace("'ll"," will ")

  # removing html tag

  q=BeautifulSoup(q,"html.parser")
  q=q.get_text()

  # remove punctuation

  pattern=re.compile("\W")
  q=re.sub(pattern," ",q)

  return q


def query_point_creator(q1,q2):

  input_query=[]

  q1=Preprocess(q1)
  q2=Preprocess(q2)

  # fetch basic features

  # len of question
  input_query.append(len(q1))
  input_query.append(len(q2))

  # number of word
  input_query.append(len(q1.split(" ")))
  input_query.append(len(q2.split(" ")))

  # common word count
  input_query.append(test_common_words(q1,q2))

  # total_words
  input_query.append(test_total_words(q1,q2))

  # word share
  input_query.append(round(test_common_words(q1,q2)/test_total_words(q1,q2),2))


  # fetch token features

  token_features=test_fetech_token_features(q1,q2)

  input_query.extend(token_features)

  # fetch length features

  length_features=test_fetech_length_features(q1,q2)
  input_query.extend(length_features)

  # fetch fuzzy features

  fuzzy_features=test_fetch_fuzzy_features(q1,q2)
  input_query.extend(fuzzy_features)

  # applying word2vec model

  q1_split=q1.split()
  q2_split=q2.split()

  doc=[word_vec.wv[word] if word in word_vec.wv.index_to_key else np.zeros(word_vec.vector_size) for word in q1_split ]
  if len(doc)==0:
    q1_vec=np.zeros(word_vec.vector_size)
  else:
    q1_vec=np.mean(doc,axis=0)

  doc=[word_vec.wv[word] if word in word_vec.wv.index_to_key else np.zeros(word_vec.vector_size) for word in q2_split ]
  if len(doc)==0:
    q2_vec=np.zeros(word_vec.vector_size)
  else:
    q2_vec=np.mean(doc,axis=0)

  return np.hstack((np.array(input_query),q1_vec,q2_vec))


