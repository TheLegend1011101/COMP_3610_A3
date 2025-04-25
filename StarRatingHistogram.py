from collections import Counter
from datasets import load_from_disk, concatenate_datasets
import numpy as np
datasets = []
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
bins = [1, 2, 3, 4, 5, 6]  # 1–5 stars
total_hist = np.zeros(len(bins) - 1, dtype=int)

for category in VALID_CATEGORIES:
    dataset = load_from_disk(fr"C:\Users\Jaheim Caesar\Downloads\cleaned_data_a3\merged_{category}")  # your actual path
    ratings = np.array(dataset['rating'])
    hist, _ = np.histogram(ratings, bins=bins)
    total_hist += hist
    del dataset  # release memory


# Print final histogram
for i, count in enumerate(total_hist, 1):
    print(f"{i} stars: {count}")
# merged_ds = load_from_disk(r"C:\Users\Jaheim Caesar\Downloads\cleaned_data_a3\merged_All_Beauty")
# ratings = merged_ds['rating']  # or whatever your column is called
# hist = Counter(ratings)
import numpy as np

# ratings = np.array(merged_ds['rating'])
# hist, bins = np.histogram(ratings, bins=[1, 2, 3, 4, 5, 6])  # 1–5 stars

# for i, count in enumerate(hist, 1):
#     print(f"{i} stars: {count}")


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Plot the histogram
plt.figure(figsize=(8, 5))
plt.bar(range(1, 6), hist, tick_label=[str(i) for i in range(1, 6)], color='skyblue', edgecolor='black')

# Axis labels and title
plt.xlabel('Star Rating')
plt.ylabel('Number of Reviews')
plt.title('Star Rating Histogram')

# Ensure no scientific notation
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

plt.tight_layout()
plt.show()