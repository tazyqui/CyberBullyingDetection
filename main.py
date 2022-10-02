# External
import nltk
import matplotlib

# Internal
from src import download, FD, analysis, words

if __name__ == "__main__":
  # Download nltk data
  print("Grabbing NLTK Values...", end=' ')

  nltk.download([
    "stopwords", "names", "punkt", "vader_lexicon",
    "averaged_perceptron_tagger", "wordnet", "omw-1.4", "words"
  ],
                quiet=True)
  print("Done.\nDownloading Data...", end=' ')

  # Get Texts from CSV
  pos_texts, neg_texts = download.get_data('data/kaggle_parsed_dataset.csv')

  # Convert Data into Frequency Distribution Tables
  print("Done.\nConverting...", end=' ')
  pos_words, neg_words = words.text_to_words(pos_texts, neg_texts)
  pos_FD = FD.getFD(pos_words)
  neg_FD = FD.getFD(neg_words)
  pos_FD, neg_FD = FD.remove_common(pos_FD, neg_FD)
  top_neg = list(map(lambda x: x[0], neg_FD.most_common(110)))
  print("Done.\nAnalyzing...", end=' ')

  #Create Features
  features = analysis.create_list_of_features(pos_texts, neg_texts, top_neg)
  print("Done.\n\n")

  # Train the Model
  print("Training...", end=' ')
  classifier = analysis.train_model(features)
  print("Done.")

  # Test the Model
  print("Model is ready to be used!")
  while True:
    text = input("\nPlease input a line of text: ")
    text = download.cleaner(text)  # Clean text for analysis

    features = analysis.extract_features(text, top_neg)

    print("\n\nFEATURES")
    print(features)
    print(classifier.classify(features))


# Testing
#print("\n\nNEGATIVE FLAGGED WORDS")
#neg_FD.tabulate(110)

#print("\n\nPOSITIVE FLAGGED WORDS")
#pos_FD.tabulate(110)
