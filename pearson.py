import random
from datasets import load_from_disk
import numpy as np
from scipy.stats import pearsonr

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
# Load one large dataset (or concatenate some if needed)
# dataset = load_from_disk(r"C:\Users\Jaheim Caesar\Downloads\cleaned_data_a3\merged_Books")

all_lengths = []
all_ratings = []

for category in VALID_CATEGORIES:
    try:
        dataset = load_from_disk(fr"C:\Users\Jaheim Caesar\Downloads\cleaned_data_a3\merged_{category}")
        sample_size = int(len(dataset) * 0.05)
        if sample_size == 0:
            continue
        
        sample = dataset.shuffle(seed=42).select(range(sample_size))
        
        # lengths = [len(str(review)) for review in sample['review_body']]
        lengths = sample['review_length']
        ratings = sample['rating']

        all_lengths.extend(lengths)
        all_ratings.extend(ratings)

        del dataset, sample  # Free memory
    except Exception as e:
        print(f"Failed to process {category}: {e}")

# Now compute Pearson correlation
if all_lengths and all_ratings:
    corr, p_value = pearsonr(all_lengths, all_ratings)
    print(f"Pearson correlation: {corr:.4f}")
    print(f"P-value: {p_value:.4e}")
    if abs(corr) < 0.2:
        interpretation = "Very weak or no linear correlation"
    elif abs(corr) < 0.5:
        interpretation = "Weak correlation"
    elif abs(corr) < 0.7:
        interpretation = "Moderate correlation"
    else:
        interpretation = "Strong correlation"

    print(f"Interpretation: {interpretation}")
else:
    print("No data to compute correlation.")
