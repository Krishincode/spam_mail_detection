# Spam Email Detection (pyhton)

This project is a simple machine learning program that detects whether an email message is spam or not. It uses a dataset of labeled emails to train a model and then predicts the category of new messages.

The program converts email text into numerical features using `CountVectorizer` and trains a Naive Bayes classifier to learn common patterns found in spam and normal emails.

## Requirements

* Python 3
* pandas
* scikit-learn

Install the required libraries:

pip install pandas scikit-learn

## How to Run

1. Make sure the dataset file `spam_data.csv` is in the same folder.
2. Run the Python script:

python spam_detector.py

The program will train the model, show the accuracy, and predict whether a sample email is spam or not.

## Purpose

This project was created to practice basic machine learning concepts such as text processing, model training, and spam classification.
