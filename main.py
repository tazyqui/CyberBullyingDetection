# External
import nltk
import matplotlib
from nltk.sentiment import SentimentIntensityAnalyzer
# Internal
from src import download, FD

if __name__ == "__main__":
  # Download nltk data
  print("Grabbing NLTK Values...", end=' ')
  nltk.download([
    "stopwords", "names", "punkt", "vader_lexicon",
    "averaged_perceptron_tagger", "wordnet", "omw-1.4", "words"
  ],
                quiet=True)
  print("Done.")

  # Get Data from CSV
  print("Downloading...", end=' ')
  pos, neg = download.get_data('data/kaggle_parsed_dataset.csv')
  print("Done.")

  # Convert Data into Frequency Distribution Tables
  print("Converting...", end=' ')
  pos_FD = FD.getFD(pos)
  neg_FD = FD.getFD(neg)
  pos_FD, neg_FD = FD.remove_common(pos_FD, neg_FD)
  print("Done.")

  # Train Model

  

  # Testing
  neg_FD.tabulate(10)
  pos_FD.tabulate(10)

  #sia = SentimentIntensityAnalyzer()

  #for ind in range(2):
  #print(pos_FD[ind])
  #print(sia.polarity_scores(dataset[ind][0]))
