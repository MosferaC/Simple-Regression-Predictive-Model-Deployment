import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('hiring.csv')

dataset['Experience'].fillna(0, inplace = True)

dataset['Test_score'].fillna(dataset['Test_score'].mean(), inplace = True)

X = dataset.iloc[:,:3]

# Converting words to integer value
def convert_to_int(word):
    
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4,
                 'five':5, 'six':6, 'seven':7, 'eight':8,
                 'nine':9, 'ten':10, 'eleven':11, 0:0}
    return word_dict[word]

X['Experience'] = X['Experience'].apply(lambda x: convert_to_int(x))

y = dataset.iloc[:, -1]

#Splitting training and test data
#Since we have a very samll dataset, we will train out model with all available data

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

# FItting model to the trained data
regressor.fit (X,y)

#Dumpong model to the disk
pickle.dump(regressor, open('model.pkl', 'wb'))


# Loading model to compare the results
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[2,9,6]]))

