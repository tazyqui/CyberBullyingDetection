import nltk

# Remove Common Words from Frequency Distribution Tables
def remove_common(pos, neg):  
  common_set = set(pos).intersection(neg)

  for word in common_set:
    del pos[word]
    del neg[word]
    
  return pos, neg

# Convert Data to Frequency Distribution Table
def getFD(data):
  FD = nltk.FreqDist(data)
  return FD
  