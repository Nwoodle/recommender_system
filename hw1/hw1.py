#%%
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import scipy as sp

data_filename = "amazon_reviews_us_Gift_Card_v1_00.tsv"

#%% [markdown]

# # Homework 1 - Renjie Zhu - A53266114

# Data parsing using ```csv.DictReader```.

#%%
data = []


#%%
with open(data_filename, newline="") as data_file:

    reader = csv.DictReader(data_file, delimiter="\t")

    for row in reader:
        data.append(row)


#%%
rating = {}

#%%
for ele in data:
    if ele["star_rating"] not in rating.keys():
        rating[ele["star_rating"]] = 0
    else:
        rating[ele["star_rating"]] += 1


#%% [markdown]

# 1. As shown in the above cell, 
# 5 stars : 129028,
# 4 stars : 9807,
# 3 stars : 3146,
# 2 stars : 1559,
# 1 stars : 4765,


#%%
rating_list = [(k,v) for k,v in rating.items()]
rating_list.sort()
rts, nums = zip(*rating_list)
plt.bar(rts,nums)


#%% [markdown]

# To train a predictor defined as
# $ R = \theta_0 + \theta_1 v + \theta_2 l $,
# we have
#
# $ R = \Theta \vec{x} = \begin{pmatrix} \theta_0 \\ \theta_1 \\ \theta_2 \end{pmatrix}
# \begin{pmatrix} 1 \\ v \\ l \end{pmatrix}
# $
#
# where $R$ is rating, $v$ is $1$ if verified and $0$ otherwise, and $l$ is length of
# the review.

#%%

def parse_XR(data):
    X = []
    R = []

    for ele in data:
        x = np.ones(3)
        if ele["verified_purchase"].upper() != "Y":
            x[1] = 0
        x[2] = len(ele["review_body"])

        X.append(x)
        R.append(int(ele["star_rating"]))

    X = np.array(X)
    R = np.array(R)

    return X, R


#%%
X, R = parse_XR(data)
t_3 = sp.linalg.lstsq(X, R)

#%%
print(f"We have theta_0 = {t_3[0][0]}, theta_1 = {t_3[0][1]}, theta_2 = {t_3[0][2]}.")


#%% [markdown]

# $\theta_0$ is a value very close to 5. This is obvious from the previous distribution
# where most reviews are given a five star. $\theta_1$ is a small positive number, and
# since the possible value is only 0 or 1, this doesn't mean much in this situation.
# $\theta_2$ is a even smaller number, but since the review length is usually a larger
# number than 5, this is expected. $\theta_2$ is also negative, which means that 
# the longer the review, the lower the rating. An interpretation of this is people
# tend to write a longer criticizing review for a bad purchase experience.

#%% [markdown]

# The predictor now only considers if the review is verified, so the problem becomes
#
# $R = \Theta \vec{x} = \begin{pmatrix} \theta_0 \\ \theta_1  \end{pmatrix}
# \begin{pmatrix} 1 \\ v \end{pmatrix}$


#%%
t_4 = sp.linalg.lstsq(X[:,:2],R)
print(f"We have theta_0 = {t_4[0][0]}, theta_1 = {t_4[0][1]}.")

#%% [markdown]

# If we do not consider the length of the review and focus only on if the
# purchase is verified, the final score is now more affected than in the 
# previous problem, from 0.050 to 0.168. This tells us that a verified buyer
# is more likely to give a higher rating though difference is small. It
# may be indicating that a non-verified buyer is occationally giving very
# low ratings to sabotage the product rating.


#%%

split = int(np.ceil(0.9 * len(data)))

train_set = data[:split]
test_set = data[split:]

#%%
X_t, R_t = parse_XR(train_set)
t_train = sp.linalg.lstsq(X_t[:,:2],R_t)
print(f"For the 90% training set, we have theta_0 = {t_train[0][0]}, theta_1 = {t_train[0][1]}.")


#%%

