import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

merged_data['sentiment'] = merged_data['rating'].apply(lambda x: 'Positive' if x > 3 else 'Negative')

X = merged_data['text']
y = merged_data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', max_df=0.8, min_df=5)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

logreg = LogisticRegression()
logreg.fit(X_train_tfidf, y_train)

y_pred = logreg.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

f1 = f1_score(y_test, y_pred, pos_label='Positive')
print(f'F1 Score: {f1:.4f}')

cm = confusion_matrix(y_test, y_pred, labels=['Positive', 'Negative'])
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Positive', 'Negative'])

cm_display.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
