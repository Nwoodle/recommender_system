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

data_filename = "amazon_reviews_us_Gift_Card_v1_00.tsv"


#%%
data = []


#%%
with open(data_filename, newline="") as data_file:

    reader = csv.DictReader(data_file, delimiter="\t")

    for row in reader:
        data.append(row)

#%%
print(data[0])

#%%
rating = {}

#%%
for ele in data:
    if ele["star_rating"] not in rating.keys():
        rating[ele["star_rating"]] = 0
    else:
        rating[ele["star_rating"]] += 1

#%%
rating

#%%
