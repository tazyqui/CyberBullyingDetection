import csv
import codecs
import re


# Clean Text Data Manually
def cleaner(text):
  # Set text to lowercase
  text = text.lower()

  # Remove newline from text
  text = text.replace("\\n", ' ')

  # Remove presidential names
  text = text.replace("obama", ' ')

  # Remove http links
  text = re.sub(r'http\S+', '', text)

  # Fix hex
  text = text.replace("\\xc2", ' ')
  text = text.replace("\\xa0", ' ')
  text = text.replace("\\x99", ' ')
  text = text.replace("\\x9d", ' ')
  text = text.replace("\\xe2", ' ')
  text = text.replace("\\xec", ' ')
  text = text.replace("\\xea", ' ')
  text = text.replace("\\xf4", ' ')
  text = text.replace("\\xea", ' ')

  # Fix leftover unicode
  text = text.encode('unicode_escape')
  text = codecs.unicode_escape_decode(text)[0]
  text = text.encode('utf-16', errors='surrogatepass').decode('utf-16')

  # Fix random users not parsed
  text = text.replace("dan_amd", ' ')
  text = text.replace("charliedemerjian", ' ')

  # Remove punctuation from text
  text = re.sub(r'[^\w\s]', ' ', text)

  return text


# Get and Format Data from CSV Dataset
def get_data(filename):
  pos_data = []
  neg_data = []

  with open(filename, encoding = "utf-8") as file:
    csv_reader = csv.reader(file,
                            quotechar='"',
                            delimiter=',',
                            quoting=csv.QUOTE_ALL)

    row_count = 0
    for row in csv_reader:
      # TESTING PURPOSE:
      #if row_count == 25:
        #break

      if row_count != 0:
        # Clean Text
        text = cleaner(row[1][1:-1])
        # Output Data as a Tuple of Words
        if row[0] == "0":
          pos_data.append(text)
        else:
          neg_data.append(text)

      row_count += 1

  return pos_data, neg_data
