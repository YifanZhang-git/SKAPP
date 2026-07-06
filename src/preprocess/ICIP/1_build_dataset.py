import argparse
import ast
import math
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval_utils import SPLIT_VALUES, assign_temporal_splits


DEFAULT_RAW_DIR = Path("datasets/raw_dataset/ICIP")
DEFAULT_OUTPUT_PATH = Path("datasets/ICIP/dataset.pkl")


def _safe_literal_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return [text]
        if isinstance(parsed, (list, tuple, set)):
            return list(parsed)
        return [parsed]
    return [value]


def encode_tags(values):
    word_to_id = {}
    encoded = []

    for value in values:
        words = _safe_literal_list(value)
        if not words:
            encoded.append([0])
            continue

        encoded_words = []
        for word in words:
            key = str(word)
            if key not in word_to_id:
                word_to_id[key] = len(word_to_id) + 1
            encoded_words.append(word_to_id[key])
        encoded.append(encoded_words)

    return encoded


def _safe_numeric(series, default=0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _dedupe_by_key(dataframe, key, name):
    duplicate_count = int(dataframe[key].duplicated().sum())
    if duplicate_count:
        print(f"{name}: dropped {duplicate_count} duplicate {key} rows with keep='first'.")
    return dataframe.drop_duplicates(key, keep="first").reset_index(drop=True)


def _clean_text(title, description):
    text = title.fillna("").astype(str) + " " + description.fillna("").astype(str)
    return text.str.replace(r"<a[^>]*>(.*?)</a>", "", regex=True).str.strip()


def process_meta_data(headers, img_info, popularity):
    headers = _dedupe_by_key(headers, "FlickrId", "headers")
    img_info = _dedupe_by_key(img_info, "FlickrId", "img_info")
    popularity = _dedupe_by_key(popularity, "FlickrId", "popularity")
    merged = pd.merge(headers, img_info, on="FlickrId", validate="one_to_one")
    merged = pd.merge(merged, popularity, on="FlickrId", validate="one_to_one")

    label_source = _safe_numeric(merged["Day30"])
    dataframe = pd.DataFrame({
        "image_id": merged["FlickrId"],
        "text": _clean_text(merged["Title"], merged["Description"]),
        "label": label_source.apply(lambda value: math.log2(value / 30 + 1)),
        "user_id": merged["UserId"],
        "date_posted": merged["DatePosted"],
        "date_taken": merged["DateTaken"],
        "date_crawl": merged["DateCrawl"],
        "size": merged["Size"],
        "num_sets": _safe_numeric(merged["NumSets"]).astype(int),
        "num_groups": _safe_numeric(merged["NumGroups"]).astype(int),
        "avg_group_members": _safe_numeric(merged["AvgGroupsMemb"]).astype(int),
        "avg_group_photos": _safe_numeric(merged["AvgGroupPhotos"]).astype(int),
        "tags": encode_tags(merged["Tags"]),
    })
    return dataframe


def process_user_data(dataset, users):
    user_columns = {
        "UserId": "user_id",
        "Ispro": "is_pro",
        "HasStats": "has_status",
        "Contacts": "contacts",
        "PhotoCount": "photo_count",
        "MeanViews": "mean_views",
    }
    user_data = users[list(user_columns)].rename(columns=user_columns)
    user_data = _dedupe_by_key(user_data, "user_id", "users")

    merged = pd.merge(dataset, user_data, on="user_id", how="left", validate="many_to_one")
    missing = int(merged["is_pro"].isna().sum())
    if missing:
        raise ValueError(f"Missing ICIP user metadata for {missing} rows.")

    for column in ["is_pro", "has_status", "contacts", "photo_count", "mean_views"]:
        merged[column] = _safe_numeric(merged[column]).astype(int)
    return merged


def _print_split_summary(dataframe):
    print("Temporal split by date_posted:")
    print(dataframe["split"].value_counts().reindex(SPLIT_VALUES).fillna(0).astype(int).to_string())
    for split_name in SPLIT_VALUES:
        subset = dataframe[dataframe["split"] == split_name]
        if len(subset) == 0:
            continue
        print(
            f"{split_name} date_posted range: "
            f"{subset['date_posted'].min()} - {subset['date_posted'].max()}"
        )


def build_dataset(raw_dir=DEFAULT_RAW_DIR, output_path=DEFAULT_OUTPUT_PATH):
    raw_dir = Path(raw_dir)
    users = pd.read_csv(raw_dir / "users_TRAIN.csv")
    headers = pd.read_csv(raw_dir / "headers_TRAIN.csv")
    img_info = pd.read_csv(raw_dir / "img_info_TRAIN.csv")
    popularity = pd.read_csv(raw_dir / "popularity_TRAIN.csv")

    dataset = process_meta_data(headers, img_info, popularity)
    dataset = process_user_data(dataset, users)
    dataset = assign_temporal_splits(dataset, "date_posted", tie_break_columns=["image_id"])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_pickle(output_path)
    print(
        "ICIP dataset processed: "
        f"{len(dataset)} rows, {dataset['image_id'].nunique()} unique image ids, "
        f"{dataset['user_id'].nunique()} users."
    )
    _print_split_summary(dataset)
    print(f"Saved ICIP dataset to {output_path}")
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Build the ICIP dataset pickle from raw CSV files.")
    parser.add_argument("--raw_dir", default=str(DEFAULT_RAW_DIR), help="Directory containing ICIP raw CSV files")
    parser.add_argument("--output_path", default=str(DEFAULT_OUTPUT_PATH), help="Output dataset pickle path")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    build_dataset(args.raw_dir, args.output_path)
    print(f"Runtime: {(time.time() - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
