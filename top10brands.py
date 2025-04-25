from datasets import load_from_disk
from collections import Counter
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

brand_counter = Counter()

for category in VALID_CATEGORIES:
    path = fr"C:\Users\Jaheim Caesar\Downloads\cleaned_data_a3\merged_{category}"
    if not os.path.exists(path):
        continue
    try:
        dataset = load_from_disk(path)
        if 'brand' in dataset.column_names:
            brands = dataset['brand']
            # Count only known brands
            brand_counter.update(b for b in brands if b and b != "Unknown")
        del dataset  # free memory
    except Exception as e:
        print(f"Error loading {category}: {e}")

# Get top 10 brands
top10_brands = brand_counter.most_common(10)

# Print them for backup
print("Top 10 Brands by Review Count:")
for brand, count in top10_brands:
    print(f"{brand}: {count:,}")

# Plot
brands, counts = zip(*top10_brands)
plt.figure(figsize=(10, 6))
plt.barh(brands[::-1], counts[::-1], color='lightgreen')
plt.xlabel("Review Count")
plt.title("Top 10 Brands by Total Review Count (excluding 'Unknown')")
plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
plt.tight_layout()
plt.show()
