from datasets import load_from_disk
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import joblib  # Import joblib for saving and loading the model

# Categories and ID mapping
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

category_id_map = {cat: i for i, cat in enumerate(VALID_CATEGORIES)}
brand_id_map = {}
brand_counter = 0

product_stats = {}

# Process each dataset one at a time
for category in VALID_CATEGORIES:
    try:
        print(f"Processing {category}...")

        dataset = load_from_disk(fr"C:\Users\Jaheim Caesar\Downloads\cleaned_data_a3\merged_{category}")
        category_id = category_id_map[category]
        

        sampled_data = dataset.shuffle(seed=42).select(range(int(len(dataset) * 0.0025)))
        
        for row in sampled_data:
            pid = row["parent_asin"]
            brand = row.get("brand", "UNKNOWN")
            if brand not in brand_id_map:
                brand_id_map[brand] = brand_counter
                brand_counter += 1
            brand_id = brand_id_map[brand]

            if pid not in product_stats:
                product_stats[pid] = {
                    "ratings": [],
                    "brand_id": brand_id,
                    "category_id": category_id  # Assigning category ID to the product
                }

            product_stats[pid]["ratings"].append(row["rating"])

        del dataset  # Free memory after sampling
    except Exception as e:
        print(f"Failed to process {category}: {e}")


data_rows = []
for pid, stats in product_stats.items():
    ratings = stats["ratings"]
    mean_rating = np.mean(ratings)
    total_reviews = len(ratings)
    data_rows.append([mean_rating, total_reviews, stats["brand_id"], stats["category_id"]])

df = pd.DataFrame(data_rows, columns=["mean_rating", "total_reviews", "brand_id", "category_id"])


kmeans = KMeans(n_clusters=5, random_state=42)
df["cluster"] = kmeans.fit_predict(df)

# Save the KMeans model
joblib.dump(kmeans, 'kmeans_model.pkl')
print("KMeans model saved as 'kmeans_model.pkl'.")

# Analyze clusters
print("\nCluster Analysis:\n")
for i in range(5):
    cluster_data = df[df["cluster"] == i]
    size = len(cluster_data)
    avg_rating = cluster_data["mean_rating"].mean()
    avg_reviews = cluster_data["total_reviews"].mean()
    avg_brand = cluster_data["brand_id"].mean()
    avg_cat = cluster_data["category_id"].mean()

    print(f"Cluster {i}:")
    print(f"  Size: {size}")
    print(f"  Avg. Mean Rating: {avg_rating:.2f}")
    print(f"  Avg. Total Reviews: {avg_reviews:.1f}")
    print(f"  Avg. Brand ID: {avg_brand:.1f}")
    print(f"  Avg. Category ID: {avg_cat:.1f}")
