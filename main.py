# External 
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
# Internal
from src import download


if __name__ == "__main__":
  nltk.download([
    "names",
    "stopwords",
    "state_union",
    "twitter_samples",
    "movie_reviews",
   "averaged_perceptron_tagger",
    "vader_lexicon",
    "punkt" ], quiet=True)

  # Get Data from CSV
  pos, neg = download.get_data()


  # Testing
  sia = SentimentIntensityAnalyzer()
  for ind in range(5):
    print(pos[ind][0] + "\n\n")
    #print(sia.polarity_scores(dataset[ind][0]))
  