#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 20:03:22 2018

@author: gauravpant
"""

import numpy as np
import pandas as pd

df=pd.read_csv('data/responses.csv', sep=',',header=0)

#f = open('data/responses.csv')
#csv_f = csv.reader(f)
#
#headers = []
#data = []
#
#
#for i, row in enumerate(csv_f):
#    if i == 0:
#         headers = row
#    else:
#        data.append(row)
#    
#print(headers)

# x is your dataset
# x = numpy.random.rand(100, 5)
#numpy.random.shuffle(df)
#training, test = df[:80,:], df[80:,:]

df['Smoking'] = pd.Categorical(df['Smoking'])
df['Smoking'] = df['Smoking'].cat.codes

df['Alcohol'] = pd.Categorical(df['Alcohol'])
df['Alcohol'] = df['Alcohol'].cat.codes

df['Punctuality'] = pd.Categorical(df['Punctuality'])
df['Punctuality'] = df['Punctuality'].cat.codes

df['Lying'] = pd.Categorical(df['Lying'])
df['Lying'] = df['Lying'].cat.codes

df['Internet usage'] = pd.Categorical(df['Internet usage'])
df['Internet usage'] = df['Internet usage'].cat.codes

df['Gender'] = pd.Categorical(df['Gender'])
df['Gender'] = df['Gender'].cat.codes

df['Left - right handed'] = pd.Categorical(df['Left - right handed'])
df['Left - right handed'] = df['Left - right handed'].cat.codes

df['Education'] = pd.Categorical(df['Education'])
df['Education'] = df['Education'].cat.codes

df['Only child'] = pd.Categorical(df['Only child'])
df['Only child'] = df['Only child'].cat.codes

df['Village - town'] = pd.Categorical(df['Village - town'])
df['Village - town'] = df['Village - town'].cat.codes

df['House - block of flats'] = pd.Categorical(df['House - block of flats'])
df['House - block of flats'] = df['House - block of flats'].cat.codes

#msk = np.random.rand(len(df)) < 0.6
#training = df[msk]
#other = df[~msk]
#msk2 = np.random.rand(len(other)) < 0.5
#dev = other[msk2]
#test = other[~msk2]

training_percent = 0.6
dev_test_percent = 0.2
np.random.seed(seed=None)
perm = np.random.permutation(df.index)
length = len(df.index)
training_end = int(training_percent * length)
dev_end = int(dev_test_percent * length) + training_end
training = df.loc[perm[:training_end]]
dev = df.loc[perm[training_end:dev_end]]
test = df.loc[perm[dev_end:]]
#print(training)