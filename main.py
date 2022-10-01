# External
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
# Internal
from src import download

if __name__ == "__main__":
  # Get Data from CSV
  print("Downloading...")
  pos, neg = download.get_data()
  print("Finished downloading...")

  # Testing
  sia = SentimentIntensityAnalyzer()
  for ind in range(2):
    print(pos[ind][0])
    #print(sia.polarity_scores(dataset[ind][0]))
