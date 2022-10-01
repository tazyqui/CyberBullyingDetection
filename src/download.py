import nltk
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import words as word_exists
import csv
import codecs
import re

# Clean Text Data Manually
def cleaner(text):
  # Set text to lowercase
  text = text.lower()

  # Remove newline from text
  text = text.replace("\\n", ' ')

  # Fix space hex
  text = text.replace("\\xc2", ' ')
  text = text.replace("\\xa0", ' ')

  # Fix leftover unicode
  text = text.encode('unicode_escape')
  text = codecs.unicode_escape_decode(text)[0]
  text = text.encode('utf-16', errors='surrogatepass').decode('utf-16')

  # Remove punctuation from text
  text = re.sub(r'[^\w\s]', ' ', text)

  return text


# Convert Text to List of Important Words
def nltk_cleaner(text, remove):
  # Separate each word in text
  words = word_tokenize(text)

  # Lemmatize, turn plural words into singular words
  lemma_entry = []
  lemma_word = ""
  for word, tag in pos_tag(words):
    # Remove unneeded word
    if word not in remove:
      # Lemmatize word
      if tag.startswith('N'):
        lemma_word = WordNetLemmatizer().lemmatize(word, pos='n')
      elif tag.startswith('V'):
        lemma_word = WordNetLemmatizer().lemmatize(word, pos='v')
      elif tag.startswith('J'):
        lemma_word = WordNetLemmatizer().lemmatize(word, pos='a')

      # Add only English words
      if lemma_word in word_exists.words():
        lemma_entry.append(lemma_word)

  return lemma_entry


# Get and Format Data from CSV Dataset
def get_data(filename):
  pos_data = []
  neg_data = []

  # List of unneeded english words
  unwanted = nltk.corpus.stopwords.words("english")
  unwanted.extend([w.lower() for w in nltk.corpus.names.words()])

  with open(filename) as file:
    csv_reader = csv.reader(file,
                            quotechar='"',
                            delimiter=',',
                            quoting=csv.QUOTE_ALL)

    row_count = 0
    for row in csv_reader:
      # TESTING PURPOSE:
      print(row_count)
      if row_count == 100:
        break
      
      if row_count == 8000/2:
        print("Almost there...", end=' ')

      if row_count != 0:
        # Clean Text
        text = cleaner(row[1][1:-1])

        # Convert to list of important words
        words = nltk_cleaner(text, unwanted)

        # Output Data as a Tuple of Words
        if int(row[0]) == 0:
          pos_data += words
        else:
          neg_data += words

      row_count += 1

  return pos_data, neg_data