#Done by Ahmed Amin Abdellatif Elmeligi

#The necessary imports:
import tensorflow as tf
from tensorflow import keras
import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path
from keras import regularizers
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer #To use bag of words model.
from sklearn.model_selection import train_test_split #To split data.
from sklearn.naive_bayes import MultinomialNB #The prediction model that will be trained and tested.
import seaborn as sns
from sklearn.metrics import accuracy_score


#STEP 1 - LOADING THE DATA AND PRE-PROCESSING:
##############################################################################################################################################################################
dataset = pd.read_csv('\\Users\\Ahmed\\Desktop\\Practicecoding\\Mypythonprojects\\DeepLearningProjects\\NLP-PNReviews-20022537\\Dataset\\Restaurant reviews.csv') #change this path accordingly.

print(dataset.info())
print(dataset.describe())
dataset.drop_duplicates() #removing any duplicates, if there are any.
dataset.drop(index=dataset[dataset['Rating']=='Like'].index,inplace=True) #Remove the anomaly of 'like' values being within the rating column.
dataset['Rating'] = dataset['Rating'].apply(float) #Change object values into float.
dataset['Rating'] = [1 if i >= 3 else 0 for i in dataset['Rating']] #Changes all values greater than or equal to 3 into 1 (positive reviews), while anything lower than 3 is changed to 0 (negative reviews).
dataset['Rating'].unique()
print(dataset['Rating'].head())



#Removing all \n and \t tags, numbers, data that isn't test, and whitespaces:
stop = set(stopwords.words('english')) #the stopwords are common words that search engines skip over to save space and computational power during the search.
def cleantext(text,stop=stop):
    if not isinstance(text,str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'\\[nt]*',' ',text) #removes \n and \t tags.
    text = re.sub(r'[^A-Za-z\s]',' ',text) #removes numbers and data that is not textual.
    text = re.sub(r'[\s+]',' ',text) #removes whitespaces.
    text = " ".join([x for x in text.split(' ') if x not in stop])#removes the stopwords from text.
    return text
dataset['Review'] = dataset['Review'].apply(cleantext)



#Feature extraction using bag of words; as in the string review data is transformed into numeric review data (Fixed-length vectors of the word counts):
CV = CountVectorizer(max_features= 5000) #Max_features parameter builds a vocabulary that only considers the n amount of top max_features inputted; And those top features are ordered by how frequently they are repeated. In this case, a total of 5000 features make up our overall vocabulary.
#The higher the max features the more accurate the model's prediction becomes; until you hit a plateau that is. in this case, 5000 is that upper limit of max features before accuracy stagnates then starts to deteriorate.
X = CV.fit_transform(dataset['Review']).toarray() #A vocabulary of 5000 features from the review data is extracted, learned and then the data is transformed into fixed-length vectors of the word counts.
Y = dataset['Rating'].values




#STEP 2 - BUILDING MODEL + TRAINING MODEL:
##########################################################################################################################################################################################
#Splitting the data into x and y train, as well as x and y test:
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 0) #30% of the data has been set aside as test data, while the rest is used for training.

#Training Naives bayes model on x and y train data:
Multinomialclassifier = MultinomialNB()#Multinomial naive bayes model.
Multinomialclassifier.fit(X_train, y_train)




#STEP 3 - TESTING AND EVALUATING THE MODEL:
###########################################################################################################################################################################################################################################
#Predicting whether a restaurant review is positive (1) or negative(0) by inputting x test into our model, so it can predict the labels (y test).
y_pred = Multinomialclassifier.predict(X_test)


#A visual representation of the confusion matrix using the seaborn library and matplotlib:
review_categories = ['Negative', 'Positive']
confusion = confusion_matrix(y_test, y_pred) #The confusion matrix showcases the wrong and correct predictions for the total reviews in either the postive or negative reviews in the dataset.
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=review_categories, yticklabels=review_categories)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()



#Accuracy score for predictions vs actual labels:
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy score: {accuracy}") #90.56% accuracy achieved and maintained.
