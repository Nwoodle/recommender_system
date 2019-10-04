# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'hw1'))
	print(os.getcwd())
except:
	pass

#%%
import csv
import matplotlib.pyplot as plt

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


#%%
