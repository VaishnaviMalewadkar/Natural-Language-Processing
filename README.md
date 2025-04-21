# Natural-Language-Processing
A collection of NLP projects covering text preprocessing, sentiment analysis, topic modeling, and model training using traditional and deep learning techniques.
Fake Job Posting Detection using NLP

#This project applies Natural Language Processing techniques to detect fake job postings from a dataset. It includes data cleaning, EDA, feature extraction (TF-IDF), and model building using machine learning classifiers like Logistic Regression and Random Forest.

#Project Overview
Job fraud is an increasing concern in the online job market. This project leverages NLP and ML techniques to identify fake job postings based on job description content and other metadata.

#Tools & Libraries
Python, Pandas, Numpy
Matplotlib, Seaborn
Scikit-learn
NLTK

#Objective
To classify job postings as real or fake based on the textual and categorical features provided.

#Features
Text preprocessing (cleaning, stopword removal, lemmatization)
Exploratory Data Analysis with visualizations
Feature extraction using TF-IDF
Model training and evaluation
Classification report and accuracy metrics

#Tools & Libraries
Python, Pandas, Numpy
Matplotlib, Seaborn
Scikit-learn
NLTK
Objective

#Dataset
Source: Kaggle - Fake Job Postings Dataset
Contains job titles, descriptions, locations, company info, and labels (fraudulent or not).
Technologies Used

#Language: Python
#Libraries:
Data: pandas, numpy
NLP: nltk, re
#Visualization: matplotlib, seaborn
ML: scikit-learn

#Project Workflow
Data Cleaning: Remove nulls, handle categorical values, clean text.
Text Preprocessing: Lowercasing, removing punctuation, stopwords, and lemmatization.
Exploratory Data Analysis: Word clouds, frequency plots, label distribution.
Feature Engineering: TF-IDF vectorization for text, encoding for categorical features.
Model Building:
Logistic Regression
Random Forest
Naive Bayes

#Evaluation: Accuracy, Precision, Recall, F1-score, Confusion Matrix.

#Model Evaluation
Multiple classifiers evaluated on standard metrics.
Confusion matrices and classification reports included for performance comparison.

#Results
Achieved high accuracy in classifying fake job postings.
Logistic Regression and Random Forest showed strong performance on TF-IDF features.

#Conclusion
This project demonstrates how NLP and machine learning can be used to combat online fraud by detecting suspicious job listings. It can be extended to real-world applications like automated job board filtering.
