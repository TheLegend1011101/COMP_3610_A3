from datasets import load_from_disk
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# Initialize storage
rating_sum_by_year = defaultdict(float)
count_by_year = defaultdict(int)
# Load your dataset
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
for category in VALID_CATEGORIES:
    dataset = load_from_disk(fr"C:\Users\Jaheim Caesar\Downloads\cleaned_data_a3\merged_{category}")  # your actual path
    # Process in batches to stay memory-safe
    # for batch in dataset.iter(batch_size=500000):
    #     for rating, year in zip(batch["rating"], batch["year"]):
    #         if year:  # avoid null years
    #             rating_sum_by_year[year] += rating
    #             count_by_year[year] += 1
    for rating, year in zip(dataset["rating"], dataset["year"]):
        if year:
            rating_sum_by_year[year] += rating
            count_by_year[year] += 1

# Compute average rating per year
avg_rating_by_year = {
    year: rating_sum_by_year[year] / count_by_year[year]
    for year in rating_sum_by_year
}

# Sort years
sorted_years = sorted(avg_rating_by_year)

# Save output to text file
output_path = "avg_rating_by_year.txt"
with open(output_path, "w") as f:
    for year in sorted_years:
        line = f"{year}: {avg_rating_by_year[year]:.3f}"
        print(line)
        f.write(line + "\n")

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(sorted_years, [avg_rating_by_year[y] for y in sorted_years], marker='o', color='green')
plt.xlabel("Year")
plt.ylabel("Average Star Rating")
plt.title("Average Star Rating per Year")
plt.grid(True)
plt.xticks(sorted_years)
plt.ylim(0, 5)
plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
plt.tight_layout()
plt.show()