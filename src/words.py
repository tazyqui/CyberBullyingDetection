import nltk
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

# Helper Function to Lemmatize List of Words
def lemma(word_tag):
  if re.search(r'\d', word_tag[0]) or len(word_tag[0]) <= 3:
    return None
  elif word_tag[1].startswith('N'):
    return WordNetLemmatizer().lemmatize(word_tag[0], pos='n')
  elif word_tag[1].startswith('V'):
    return WordNetLemmatizer().lemmatize(word_tag[0], pos='v')
  elif word_tag[1].startswith('J'):
    return WordNetLemmatizer().lemmatize(word_tag[0], pos='a')


# Convert Text to List of Important Words
def nltk_cleaner(text, remove):
  # Separate each word in text
  words = word_tokenize(text)

  # Lemmatize, turn plural words into singular words
  lemma_entry = list(map(lemma, pos_tag(words)))

  # Remove unneeded words
  lemma_entry = list(set(lemma_entry).difference(remove))
  lemma_entry = filter(None, lemma_entry)

  return lemma_entry


# Get and Format Data from CSV Dataset
def text_to_words(pos_texts, neg_texts):
  pos_words = []
  neg_words = []
  # List of unneeded english words
  unwanted = nltk.corpus.stopwords.words("english")
  unwanted.extend([w.lower() for w in nltk.corpus.names.words()])

  for sentence in pos_texts:
    # Convert to list of important words
    words = nltk_cleaner(sentence, unwanted)

    # Output Data as a Tuple of Words
    pos_words += words
    
  for sentence in neg_texts:
    # Convert to list of important words
    words = nltk_cleaner(sentence, unwanted)

    # Output Data as a Tuple of Words
    neg_words += words

  return pos_words, neg_words
