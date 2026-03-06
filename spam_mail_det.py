# Spam Email Detection using Naive Bayes

# Import required libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# Step 1: Load the dataset
# -----------------------------
data = pd.read_csv("spam_data.csv")

# Email text and labels
emails = data["text"]
labels = data["label"]


# -----------------------------
# Step 2: Convert text into numbers
# Machine learning models cannot understand text directly
# -----------------------------
vectorizer = CountVectorizer()
email_features = vectorizer.fit_transform(emails)


# -----------------------------
# Step 3: Split dataset
# 80% training and 20% testing
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    email_features, labels, test_size=0.2, random_state=42
)


# -----------------------------
# Step 4: Train the model
# -----------------------------
spam_model = MultinomialNB()
spam_model.fit(X_train, y_train)


# -----------------------------
# Step 5: Test the model
# -----------------------------
predictions = spam_model.predict(X_test)


# -----------------------------
# Step 6: Evaluate performance
# -----------------------------
print("Model Accuracy:", accuracy_score(y_test, predictions))
print("\nDetailed Report:\n")
print(classification_report(y_test, predictions))


# -----------------------------
# Step 7: Try a custom email
# -----------------------------
test_email = ["Congratulations! You won a free ticket!"]

test_email_vector = vectorizer.transform(test_email)
result = spam_model.predict(test_email_vector)

print("\nPrediction for the new email:", result[0])
