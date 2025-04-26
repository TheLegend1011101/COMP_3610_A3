import os
from datasets import load_from_disk, Dataset
import pandas as pd
from surprise import Reader, Dataset as SurpriseDataset, SVD
from surprise.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import random
import numpy as np

def filter_users_by_review_count(df: pd.DataFrame, min_reviews: int = 1) -> pd.DataFrame:
    """Filters out users with fewer than min_reviews."""
    review_counts = df['user_id'].value_counts()
    valid_users = review_counts[review_counts >= min_reviews].index
    filtered_df = df[df['user_id'].isin(valid_users)]
    return filtered_df

def train_als_model(train_set, n_factors=50, n_epochs=10, random_state=42):
    """Trains an ALS-like model (SVD in Surprise)."""
    model = SVD(n_factors=n_factors, random_state=random_state)
    model.fit(train_set)
    return model

def evaluate_model(model, test_set):
    """Evaluates the model using RMSE."""
    predictions = model.test(test_set)
    rmse = np.sqrt(mean_squared_error([pred.r_ui for pred in predictions], [pred.est for pred in predictions]))
    return rmse, predictions

def get_top_n_recommendations(model, df: pd.DataFrame, user_id: str, n=5):
    """Gets the top N recommendations for a given user."""
    user_items = df[df['user_id'] == user_id]['product_id'].unique()
    all_items = df['product_id'].unique()
    items_to_predict = np.setdiff1d(all_items, user_items)

    predictions = [model.predict(user_id, item_id) for item_id in items_to_predict]
    top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]
    return top_n

if __name__ == "__main__":
    base_input_dir = r"C:\Users\darri\OneDrive - The University of the West Indies, St. Augustine\Year3\COMP 3610\A3\amazon_reviews_datasets_reloaded"
    valid_categories_full = [] 
    valid_categories_sampled = [ "All_Beauty", "Amazon_Fashion", "Appliances", "Arts_Crafts_and_Sewing", "Automotive",
    "Baby_Products", "Beauty_and_Personal_Care", "Books", "CDs_and_Vinyl",
    "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry", "Digital_Music", "Electronics",
    "Gift_Cards", "Grocery_and_Gourmet_Food", "Handmade_Products", "Health_and_Household",
    "Health_and_Personal_Care", "Home_and_Kitchen", "Industrial_and_Scientific", "Kindle_Store",
    "Magazine_Subscriptions", "Movies_and_TV", "Musical_Instruments", "Office_Products",
    "Patio_Lawn_and_Garden", "Pet_Supplies", "Software", "Sports_and_Outdoors",
    "Subscription_Boxes", "Tools_and_Home_Improvement", "Toys_and_Games", "Video_Games", "Unknown" ] # These are the base names

    print("--- Processing ALS Recommender for Each Category ---")

    all_categories_to_process = list(set(valid_categories_full + valid_categories_sampled))

    for category in all_categories_to_process:
        input_path_full = os.path.join(base_input_dir, f"merged_{category}")
        input_path_sampled = os.path.join(base_input_dir, f"merged_{category}_single_proc_sampled")
        input_path = None
        is_sampled = False

        if os.path.exists(input_path_sampled):
            input_path = input_path_sampled
            is_sampled = True
            print(f"\n--- Processing Category (Sampled): {category} ---")
        elif os.path.exists(input_path_full):
            input_path = input_path_full
            print(f"\n--- Processing Category (Full): {category} ---")
        else:
            print(f"\n--- Skipping Category: {category} - No processed data found ---")
            continue

        try:
            loaded_dataset = load_from_disk(input_path)
            df = loaded_dataset.to_pandas()

            # 1. Data Setup
            df.rename(columns={'asin': 'product_id'}, inplace=True)
            df_filtered = filter_users_by_review_count(df)
            print(f"  Number of reviews after filtering users: {len(df_filtered)}")

            if len(df_filtered) == 0:
                print("  Skipping category due to no reviews after filtering.")
                continue

            # 2. Split Data
            reader = Reader(rating_scale=(1, 5))
            surprise_data = SurpriseDataset.load_from_df(df_filtered[['user_id', 'product_id', 'rating']], reader)
            train_set, test_set = train_test_split(surprise_data, test_size=0.2, random_state=42)

            # 3. ALS (using SVD)
            als_model = train_als_model(train_set)

            # 4. Evaluation
            rmse, predictions = evaluate_model(als_model, test_set)
            print(f"  RMSE on the test set for {category} (Sampled: {is_sampled}): {rmse:.4f}")

            # 5. Demo
            test_df = pd.DataFrame([(pred.uid, pred.iid, pred.r_ui) for pred in predictions],
                                    columns=['user_id', 'product_id', 'actual_rating'])
            random_users = test_df['user_id'].unique()
            if len(random_users) >= 3:
                random_sample_users = random.sample(list(random_users), 3)
                print("  Top 5 Recommendations for Random Users:")
                for user_id in random_sample_users:
                    top_recommendations = get_top_n_recommendations(als_model, df_filtered, user_id)
                    print(f"    User ID: {user_id}")
                    for rec in top_recommendations:
                        print(f"      Product ID: {rec.iid}, Predicted Rating: {rec.est:.4f}")
            elif len(random_users) > 0:
                random_sample_users = random.sample(list(random_users), len(random_users))
                print("  Top 5 Recommendations for Available Random Users:")
                for user_id in random_sample_users:
                    top_recommendations = get_top_n_recommendations(als_model, df_filtered, user_id)
                    print(f"    User ID: {user_id}")
                    for rec in top_recommendations:
                        print(f"      Product ID: {rec.iid}, Predicted Rating: {rec.est:.4f}")
            else:
                print("  Not enough users in the test set to show recommendations.")

        except FileNotFoundError:
            print(f"  Error: Dataset not found at {input_path}")
        except Exception as e:
            print(f"  An error occurred: {e}")

    print("\n--- ALS Recommender Processing Complete for All Categories ---")