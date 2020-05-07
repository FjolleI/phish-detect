
import pandas as pd

#Collection of Data
legitimate_urls = pd.read_csv("../extracted_csv_files/legitimate-urls.csv")
phishing_urls = pd.read_csv("../extracted_csv_files/phishing-urls.csv")

print(len(legitimate_urls))
print(len(phishing_urls))

#Data PreProcessing
urls = legitimate_urls.append(phishing_urls)

print(len(urls))
print(urls.columns)

#Removing Unnecessary columns
urls = urls.drop(urls.columns[[0,3,5]],axis=1)
print(urls.columns)

#Shuffling the rows in the dataset so that when splitting the train and test set are equally distributed
urls = urls.sample(frac=1).reset_index(drop=True)

#Removing class variable from the dataset
urls_without_labels = urls.drop('label',axis=1)
urls_without_labels.columns
labels = urls['label']

#Splitting the data into train data and test data
import random
random.seed(100)
from sklearn.model_selection import train_test_split
data_train, data_test, labels_train, labels_test = train_test_split(urls_without_labels, labels, test_size=0.20, random_state=100)
print(len(data_train),len(data_test),len(labels_train),len(labels_test))
print(labels_train.value_counts())
print(labels_test.value_counts())

#Random Forest
from flask import render_template
from sklearn.ensemble import RandomForestClassifier
RFmodel = RandomForestClassifier()
RFmodel.fit(data_train,labels_train)
rf_pred_label = RFmodel.predict(data_test)

from sklearn.metrics import confusion_matrix,accuracy_score
cm2 = confusion_matrix(labels_test,rf_pred_label)
print(cm2)
score=accuracy_score(labels_test,rf_pred_label)
score=round(score*100,2)
print(score)
print(accuracy_score(labels_test,rf_pred_label))

#Saving the model to a file
import pickle
file_name = "RandomForestModel.sav"
pickle.dump(RFmodel,open(file_name,'wb'))