import os
from datasets import load_from_disk
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Your category names and paths
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

category_counts = {}

for category in VALID_CATEGORIES:
    path = fr"C:\Users\Jaheim Caesar\Downloads\cleaned_data_a3\merged_{category}"
    if not os.path.exists(path):
        continue
    try:
        dataset = load_from_disk(path)
        category_counts[category] = len(dataset)
        del dataset  # free memory
    except Exception as e:
        print(f"Error loading {category}: {e}")

# Sort and select top 10
top10 = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]

# Print top 10 so you can rerun easily
print("Top 10 categories by review count:")
for category, count in top10:
    print(f"{category}: {count:,}")

# Plot
categories, counts = zip(*top10)
plt.figure(figsize=(10, 6))
bars = plt.barh(categories[::-1], counts[::-1], color='skyblue')
plt.xlabel("Review Count")
plt.title("Top 10 Categories by Total Review Count")

# Turn off scientific notation
plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

plt.tight_layout()
plt.show()
