# Data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Options
pd.options.display.width = 200
test_output = True

# Loading Data
print('Loading data')
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')
combine = [train_df, test_df]

# Testing
if test_output:

  print('Testing')

  # Output df titles, shape, info and head
  #print(train_df.columns.values)
  #print('_'*40)
  #print train_df.shape
  #print('_'*40)
  print(train_df.head())
  #print('_'*40)
  #print(train_df.info())
  #print('_'*40)
  #print(train_df.describe())
  print('_'*40)

  # Output Survival vs tables
  print(train_df[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean().sort_values('Survived', ascending=False))
  print('_'*40)
  print(train_df[['Sex','Survived']].groupby(['Sex'], as_index=False).mean().sort_values('Survived', ascending=False))
  print('_'*40)
  print(train_df[['Embarked','Survived']].groupby(['Embarked'], as_index=False).mean().sort_values('Survived', ascending=False))
