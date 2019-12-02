import numpy as np 
import pandas as pd 
import pprint
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer 

pp = pprint.PrettyPrinter(indent=4)

'''
Preprocessing:
	- Keep only closed case
	- Keep case with district data

Features:
	- Service name: service_name, onehot
	- Total population: total_pop2017
	- White population: white_pop2017
	- African American population: afram_pop2017
	- Hispanic population population: hisp_pop2017
	- Case origion: case_origin, onehot
	- Description sentiment: feature sentiment analysis, homebrewed
        - neg, neu, com, pos
    - Average income: avg_agi
'''

def train_test_validation(closed_data, train_frac=0.8):
    """ 8:1:1 train, text ,validation split """
    train = closed_data.sample(frac=train_frac)
    remain = closed_data.drop(train.index)
    test = remain.sample(frac=0.5)
    validation = remain.drop(test.index)
    return train, test, validation

def data_preprocessing(filename="get_it_done.csv", threshold=1.0):
    """ return processed dataset (datframe) for classifier """
    data = pd.read_csv(filename)
    data = data.loc[data["status"].isin(["Closed"])]
    data = data.loc[data["district"].isin([1,2,3,4,5,6,7,8,9])]

    data["avg_agi"].replace('', np.nan, inplace=True)
    data.dropna(subset=["avg_agi"], inplace=True)

    # reference:

    # Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious 
    # Rule-based Model for Sentiment Analysis of Social Media 
    # Text. Eighth International Conference on Weblogs and Social
    #  Media (ICWSM-14). Ann Arbor, MI, June 2014.

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

    column_lst = ["total_pop2017", "white_pop2017", "afram_pop2017", "hisp_pop2017", "avg_agi", "neg", "neu", "com", "pos"]
    onehot_lst = ["service_name", "case_origin"]
    prep_data = data[column_lst]


    # Onehot Process
    onehot_data = data[onehot_lst]
    categorize_data = pd.get_dummies(onehot_data, prefix=onehot_lst)
    dataset = pd.concat([prep_data, categorize_data], axis=1)

    # label 
    logistic_label = data["lncase_age_days"] <= threshold
    dataset["logis_label"] = logistic_label.astype(int)

    return dataset

def data_label(dataset):
    """separate data and label"""
    label = np.array(dataset.loc[:, dataset.columns=="logis_label"])
    data = np.array(dataset.loc[:, dataset.columns!="logis_label"])
    
    return data, label