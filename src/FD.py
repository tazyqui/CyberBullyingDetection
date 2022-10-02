import nltk


# Remove Common Words from Frequency Distribution Tables
def remove_common(pos, neg):
  common_set = set(pos).intersection(neg)

  for word in common_set:
    # Remove words with high frequencies in both
    #if pos[word] > 20 and neg[word] > 20:
    #  del neg[word]
    #  del pos[word]

    # Remove word if more common in other list
    if neg[word] > pos[word]:
      del pos[word]
    else:
      del neg[word]

  return pos, neg


# Convert Data to Frequency Distribution Table
def getFD(data):
  FD = nltk.FreqDist(data)
  return FD
