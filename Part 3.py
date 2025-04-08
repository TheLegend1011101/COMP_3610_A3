import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


sns.set(style="whitegrid")


plt.figure(figsize=(8, 6))
sns.histplot(merged_data['rating'], bins=5, kde=False, color='blue', discrete=True)
plt.title('Star Rating Distribution (1-5)', fontsize=15)
plt.xlabel('Rating', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()


category_counts = merged_data['main_category'].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=category_counts.index, y=category_counts.values, palette="Blues_d")
plt.title('Top 10 Categories by Total Review Count', fontsize=15)
plt.xlabel('Category', fontsize=12)
plt.ylabel('Total Reviews', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.show()


brand_counts = merged_data[merged_data['details'].apply(lambda x: x.get('brand') != 'Unknown')]['details'].apply(lambda x: x.get('brand')).value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=brand_counts.index, y=brand_counts.values, palette="Set2")
plt.title('Top 10 Brands by Total Review Count', fontsize=15)
plt.xlabel('Brand', fontsize=12)
plt.ylabel('Total Reviews', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.show()


merged_data['year'] = pd.to_datetime(merged_data['timestamp'], unit='s').dt.year
avg_rating_per_year = merged_data.groupby('year')['rating'].mean()
plt.figure(figsize=(10, 6))
sns.lineplot(x=avg_rating_per_year.index, y=avg_rating_per_year.values, marker='o')
plt.title('Average Star Rating per Year', fontsize=15)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Average Rating', fontsize=12)
plt.show()


merged_data['review_length'] = merged_data['text'].apply(len)
correlation = merged_data[['review_length', 'rating']].corr(method='pearson')
print(f"Pearson Correlation between Review Length and Star Rating: {correlation.loc['review_length', 'rating']:.4f}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=merged_data['review_length'], y=merged_data['rating'], color='purple', alpha=0.5)
plt.title('Review Length vs Star Rating', fontsize=15)
plt.xlabel('Review Length', fontsize=12)
plt.ylabel('Star Rating', fontsize=12)
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(merged_data['helpful_vote'], bins=20, kde=False, color='green')
plt.title('Distribution of Helpful Vote Counts', fontsize=15)
plt.xlabel('Helpful Votes', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x='verified_purchase', y='rating', data=merged_data, palette="Set1")
plt.title('Verified Purchase vs Star Rating', fontsize=15)
plt.xlabel('Verified Purchase', fontsize=12)
plt.ylabel('Star Rating', fontsize=12)
plt.show()
