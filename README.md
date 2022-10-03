# CyberBullyingDetection  SASEHack Fall 2022
Team: Tan Nguyen, Giovanni Cornejo

Prompt:Create a solution (or improve an existing solution) that helps identify, stop, and/or remove harassment, bullying, and/or discrimination for kids ages 8-15 years old.

Solution: We made a python program using the nltk library's Sentiment Analyzer to train a model that would detect whether a text is considered bullying/harassment or not.

Data: We used 2 datasets provided by Kaggle (kaggle_parsed_dataset and twitter_parsed_dataset). https://www.kaggle.com/datasets/saurabhshahane/cyberbullying-dataset 

Result: The program works with a ~70% accuracy according to the classifier we trained. However, it's sometime too sensitive to certain words and not sensitive enough 
to other words. We think that the model could be improve with more data. However, the more data provided, the longer it takes for the model to build.

Compilation Instruction:
1. Download python (version 3.8.12 was used for this project)
2. Install nltk library.
3. You can now run the program by going to the folder that contain main.py in the command prompt and type "python main.py"
