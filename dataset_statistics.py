# Code used for generate statistics figure for the dataset
import numpy as np 
import pandas as pd
import pprint
import nltk
import matplotlib.pyplot as plt
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
pp = pprint.PrettyPrinter(indent=4)
# Load Dataset
data = pd.read_csv('get_it_done.csv')
data = data.loc[data["status"].isin(["Closed"])]
data = data.loc[data["district"].isin([1,2,3,4,5,6,7,8,9])]

# Clear uncomplete data
for zc in [92125,92134,92132,92147,92140]:
    mask = data["zipcode"] == zc
    data = data[~mask]

# Generate sentiment analysis
neg = []
neu = []
com = []
pos = []

sid = SentimentIntensityAnalyzer()
for sentence in data["public_description"]:
    if not isinstance(sentence, str):
        sentence = ""
    ss = sid.polarity_scores(sentence)
    neg.append(ss["neg"])
    neu.append(ss["neu"])
    com.append(ss["compound"])
    pos.append(ss["pos"])

data["neg"] = neg
data["neu"] = neu
data["com"] = com
data["pos"] = pos

# Print maximum case age days
print(data['case_age_days'].max())

# Data Cleaning of case_age_days attribute
data["case_age_days"].replace('', np.nan, inplace=True)
data.dropna(subset=["case_age_days"], inplace=True)

# Generate and save histogram of case type
data['case_record_type'].hist(figsize=(16,8))
plt.title('Histogram of case type')
plt.savefig('./fig/case_type_hist.png')

