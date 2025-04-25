import os
import re
import traceback
from datetime import datetime

from datasets import load_from_disk, disable_caching
disable_caching()

def extract_brand_batch(batch):
    stores = batch.get("store", [])
    details_list = batch.get("details", [])
    result = []
    for store, details in zip(stores, details_list):
        if store and store.strip():
            result.append(store.strip())
        else:
            try:
                d = eval(details) if isinstance(details, str) else details
                result.append(d.get("Brand", "Unknown").strip() if d and "Brand" in d else "Unknown")
            except:
                result.append("Unknown")
    return {"brand": result}

# def add_review_length(batch):
#     reviews = batch.get("text", [])
#     return {
#         "review_length": [len(r.split()) for r in reviews]  # must match len(reviews)
#     }
import re
word_pattern = re.compile(r'\S+')

def add_review_length(batch):
    reviews = batch.get("text", [])
    return {
        "review_length": [len(word_pattern.findall(r)) for r in reviews]
    }


def add_year(batch):
    timestamps = batch.get("timestamp", [])
    
    if not timestamps:
        return {"year": [0] * len(batch[next(iter(batch))])}

    years = []
    for ts in timestamps:
        try:
            dt = datetime.fromtimestamp(ts / 1000)
            years.append(dt.year)
        except:
            years.append(0)
    return {"year": years}

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

base_input_dir = r"C:\Users\Jaheim Caesar\Downloads\A3Data"
base_output_dir = r"C:\Users\Jaheim Caesar\Downloads\cleaned_data_a3"
os.makedirs(base_output_dir, exist_ok=True)
def main():
    for category in VALID_CATEGORIES:
        print(f"Processing category: {category}")
        meta_path = os.path.join(base_input_dir, f"raw_meta_{category}/full")
        review_path = os.path.join(base_input_dir, f"raw_review_{category}/full")

        if not os.path.exists(meta_path) or not os.path.exists(review_path):
            print(f"❌ Skipping {category}: path does not exist.")
            continue

        try:
            meta_ds = load_from_disk(meta_path)
            review_ds = load_from_disk(review_path)

            meta_ds = meta_ds.map(
                extract_brand_batch,
                batched=True,
                num_proc=4,
                desc="Extracting brands"
            )
            meta_ds = meta_ds.remove_columns(
                [col for col in meta_ds.column_names if col not in {"parent_asin", "brand", "main_category"}]
            )
            review_ds = review_ds.filter(
                lambda row: row.get("rating") in [1, 2, 3, 4, 5] and row.get("text") not in (None, ""),
                num_proc=4,
                desc="Filtering reviews"
            )

            review_ds = review_ds.map(
                add_review_length,
                batched=True,
                batch_size=2000,
                num_proc=6,
                desc="Adding review length"
            ).map(
                add_year,
                batched=True,
                num_proc=4,
                desc="Adding year"
            )

           
            meta_lookup = {row['parent_asin']: row for row in meta_ds if row.get('parent_asin')}

        
            from functools import partial

            def make_merge_meta_batch(meta_lookup):
                def merge_meta_batch(batch):
                    parent_asins = batch["parent_asin"]
                    main_categories = []
                    brands = []

                    for asin in parent_asins:
                        meta = meta_lookup.get(asin)
                        if meta:
                            main_categories.append(meta.get("main_category"))
                            brands.append(meta.get("brand", "Unknown"))
                        else:
                            main_categories.append(None)
                            brands.append("Unknown")

                    batch["main_category"] = main_categories
                    batch["brand"] = brands
                    return batch

                return merge_meta_batch

            merge_fn = make_merge_meta_batch(meta_lookup)

            review_ds = review_ds.filter(lambda x: x.get("parent_asin"), num_proc=4)
            merged = review_ds.map(
                merge_fn,
                batched=True,
                batch_size=2000,
                num_proc=6,
                remove_columns=[],
                desc="Merging review and meta"
            )
            del meta_lookup
            del meta_ds
            del review_ds
            import gc
            gc.collect()
            # Deduplicate
            seen = set()
            def dedup(row):
                key = (row["user_id"], row["asin"], row["text"])
                if key in seen:
                    return False
                seen.add(key)
                return True

            merged = merged.filter(dedup, desc="Removing duplicates")

            keep_cols = [
                'rating', 'title', 'text', 'asin', 'parent_asin',
                'user_id', 'timestamp', 'verified_purchase',
                'review_length', 'year', 'main_category', 'brand'
            ]
            merged = merged.remove_columns([col for col in merged.column_names if col not in keep_cols])

            output_path = os.path.join(base_output_dir, f"merged_{category}")
            merged.save_to_disk(output_path, num_proc=4)
            print(f"✅ Successfully saved cleaned dataset for {category}")

        except Exception as e:
            print(f"❌ Error processing {category}: {e}")
            print(traceback.format_exc())

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()