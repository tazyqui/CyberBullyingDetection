# External
import nltk

# Internal
from src import download, FD, analysis, words
import os

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
  pos_texts2, neg_texts2 = download.get_data('data/twitter_parsed_dataset.csv')
  
  pos_texts += pos_texts2
  neg_texts += neg_texts2

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
  classifier, accuracy = analysis.train_model(features)
  print("Done.")

  print("Accuracy: ", end='')
  print(accuracy)

  # Test the Model
  print("\nModel is ready to be used!")
  while True:

    text = input("\nPlease input a line of text: ")
    os.system('clear')

    print("INPUT\n" + text)
    text = download.cleaner(text)

    features = analysis.extract_features(text, top_neg)
    classification = classifier.classify(features)

    print("\nFEATURES")
    print(features)
    print("\nCLASSIFICATION")
    if(classification == "pos"):
      print("Not Bullying/Harassment")
    else:
      print("Bullying/Harassment Detected!")
    

# Testing
#print("\n\nNEGATIVE FLAGGED WORDS")
#neg_FD.tabulate(110)

#print("\n\nPOSITIVE FLAGGED WORDS")
#pos_FD.tabulate(110)
