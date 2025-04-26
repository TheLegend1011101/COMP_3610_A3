from datasets import load_from_disk, concatenate_datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd

# Load dataset
def main():
    VALID_CATEGORIES = [
        "All_Beauty", "Amazon_Fashion", "Appliances", "Arts_Crafts_and_Sewing", "Automotive",
        "Baby_Products", "Beauty_and_Personal_Care", "Books", "CDs_and_Vinyl",
        "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry", "Digital_Music", "Electronics",
        "Gift_Cards", "Grocery_and_Gourmet_Food", "Handmade_Products", "Health_and_Household",
        "Health_and_Personal_Care", "Home_and_Kitchen", "Industrial_and_Scientific", "Kindle_Store",
        "Magazine_Subscriptions", "Movies_and_TV", "Musical_Instruments", "Office_Products",
        "Patio_Lawn_and_Garden", "Pet_Supplies", "Software", "Sports_and_Outdoors",
        "Subscription_Boxes", "Tools_and_Home_Improvement", "Toys_and_Games", "Video_Games", "Unknown"
    ]
    concatdata = []
    for category in VALID_CATEGORIES:
        dataset = load_from_disk(fr"D:\Cleaned_Dataset\merged_{category}")
        dataset = dataset.shuffle(seed=42).select(range(int(len(dataset) * 0.0001)))  # Sample 2.5%
        concatdata.append(dataset)
    small_data = concatenate_datasets(concatdata)
    # Sample only 0.25% of the data
    # small_data = data.shuffle(seed=42).select(range(int(len(data) * 0.0025)))

    # Transform rating into binary sentiment
    def label_sentiment_batch(batch):
        batch['label'] = [1 if rating > 3 else 0 for rating in batch['rating']]
        return batch

    small_data = small_data.map(label_sentiment_batch, batched=True, num_proc=4, remove_columns=['rating'])

    # Extract text and labels
    texts = small_data['text']  # change 'review' to the actual text field
    labels = small_data['label']

    # Split data: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, shuffle=True)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        lowercase=True,
        tokenizer=None,  # use default tokenization
        min_df=5,
        max_df=0.8
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Logistic Regression Classifier
    clf = LogisticRegression()
    clf.fit(X_train_vec, y_train)

    # Predictions
    y_pred = clf.predict(X_test_vec)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Results
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("Confusion Matrix (TP, FP, TN, FN):")
    tn, fp, fn, tp = conf_matrix.ravel()
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support
    main()
