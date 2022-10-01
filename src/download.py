import nltk
import csv
import re


# Clean Text Data Manually
def cleaner(text):
  # Set text to lowercase
  text = text.lower()

  # Remove newline from text
  text = text.replace("\\n", ' ')

  # Remove leftover utf-8 encoding
  text = text.replace("\\\\xc2\\\\xa0", ' ')
  text = text.replace("\\xa0", ' ')

  # Remove punctuation from text
  text = re.sub(r'[^\w\s]', ' ', text)

  # Fix issue with leftover utf-8 encoding
  text = text.replace("xc2xa0", ' ')

  return text


# Convert Text to List of Important Words
def nltk_cleaner(text, remove):
  # Remove unneeded words
  words = nltk.word_tokenize(text)
  words = [word for word in words if word not in remove]
  return words


# Get and Format Data from CSV Dataset
def get_data():
  pos_data = []
  neg_data = []

  # List of unneeded english words
  unwanted = nltk.corpus.stopwords.words("english")
  unwanted.extend([w.lower() for w in nltk.corpus.names.words()])

  with open('data/kaggle_parsed_dataset.csv') as file:
    csv_reader = csv.reader(file,
                            quotechar='"',
                            delimiter=',',
                            quoting=csv.QUOTE_ALL)

    row_count = 0
    for row in csv_reader:
      if row_count != 0:
        # Clean Text
        text = cleaner(row[3][1:-1])

        # Convert to list of important words
        words = nltk_cleaner(text, unwanted)

        # Output Data as a Tuple including Text, Flag
        data_tup = (words, int(row[1]))

        # Put in corresponding list
        if int(row[1]) == 0:
          pos_data.append(data_tup)
        else:
          neg_data.append(data_tup)

      row_count += 1

  return pos_data, neg_data