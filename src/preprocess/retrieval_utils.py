import heapq
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


DEFAULT_SEED = 12
SPLIT_VALUES = ("train", "valid", "test")

DANGEROUS_FEATURES = {
    "label",
    "labellog2",
    "label_log2",
    "day30",
    "likecount",
    "like_count",
    "viewcount",
    "view_count",
    "commentnum",
    "comment_num",
    "retrievedlabel",
    "retrieved_label",
    "retrievedlabellist",
    "retrieved_label_list",
    "rrcpsilver",
    "rrcp_silver",
    "rrcpgold",
    "rrcp_gold",
    "prediction",
    "predicted",
    "output",
    "taken_timestamp",
}


def _normalized_feature_name(name):
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def _compact_feature_name(name):
    return _normalized_feature_name(name).replace("_", "")


def validate_retrieval_features(features):
    dangerous = []
    for feature in features:
        normalized = _normalized_feature_name(feature)
        compact = _compact_feature_name(feature)
        if normalized in DANGEROUS_FEATURES or compact in DANGEROUS_FEATURES:
            dangerous.append(feature)
    if dangerous:
        raise ValueError(
            "Refusing to use target, post-outcome, or derived-label columns for retrieval: "
            f"{dangerous}"
        )


def _temporal_sort_key(series, time_column):
    numeric = pd.to_numeric(series, errors="coerce")
    if not numeric.isna().any():
        return numeric

    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if parsed.isna().any():
        bad_count = int(parsed.isna().sum())
        raise ValueError(
            f"Cannot create a temporal split: {time_column} has {bad_count} missing or invalid timestamps."
        )
    return parsed.astype("int64")


def assign_temporal_splits(dataframe, time_column, tie_break_columns=None,
                           train_ratio=0.8, valid_ratio=0.1):
    if time_column not in dataframe.columns:
        raise KeyError(f"Temporal split column not found: {time_column}")

    tie_break_columns = list(tie_break_columns or [])
    missing_tie_breaks = [column for column in tie_break_columns if column not in dataframe.columns]
    if missing_tie_breaks:
        raise KeyError(f"Temporal split tie-break columns not found: {missing_tie_breaks}")

    total = len(dataframe)
    if total < 3:
        raise ValueError("At least 3 rows are required to create train/valid/test temporal splits.")
    if train_ratio <= 0 or valid_ratio <= 0 or train_ratio + valid_ratio >= 1:
        raise ValueError("Temporal split ratios must satisfy train_ratio > 0, valid_ratio > 0, and sum < 1.")

    data = dataframe.copy()
    data["_temporal_sort_key"] = _temporal_sort_key(data[time_column], time_column)
    data = data.sort_values(["_temporal_sort_key", *tie_break_columns], kind="mergesort").reset_index(drop=True)

    test_ratio = 1.0 - train_ratio - valid_ratio
    test_count = max(1, int(round(total * test_ratio)))
    valid_count = max(1, int(round(total * valid_ratio)))
    if test_count + valid_count >= total:
        test_count = 1
        valid_count = 1
    train_count = total - valid_count - test_count

    split = (
        ["train"] * train_count
        + ["valid"] * valid_count
        + ["test"] * test_count
    )
    data["split"] = split
    return data.drop(columns=["_temporal_sort_key"])


def split_and_save_pkl(input_path, train_path, valid_path, test_path, seed=DEFAULT_SEED):
    dataset = pd.read_pickle(input_path)

    train_data, valid_data = train_test_split(dataset, test_size=0.2, random_state=seed)
    valid_data, test_data = train_test_split(valid_data, test_size=0.5, random_state=seed)

    train_data.reset_index(drop=True, inplace=True)
    valid_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    train_data.to_pickle(train_path)
    valid_data.to_pickle(valid_path)
    test_data.to_pickle(test_path)
    return {
        "train": int(len(train_data)),
        "valid": int(len(valid_data)),
        "test": int(len(test_data)),
    }


def split_by_column_and_save_pkl(input_path, train_path, valid_path, test_path, split_column="split"):
    dataset = pd.read_pickle(input_path)
    if split_column not in dataset.columns:
        raise KeyError(f"Split column not found: {split_column}")

    normalized = dataset[split_column].astype(str).str.lower()
    train_data = dataset[normalized == "train"].copy()
    valid_data = dataset[normalized.isin(["valid", "val", "validation"])].copy()
    test_data = dataset[normalized == "test"].copy()

    if min(len(train_data), len(valid_data), len(test_data)) == 0:
        counts = normalized.value_counts().to_dict()
        raise ValueError(f"Invalid split column {split_column}; split counts: {counts}")

    train_data.reset_index(drop=True, inplace=True)
    valid_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    train_data.to_pickle(train_path)
    valid_data.to_pickle(valid_path)
    test_data.to_pickle(test_path)
    return {
        "train": int(len(train_data)),
        "valid": int(len(valid_data)),
        "test": int(len(test_data)),
    }


def write_split_manifest(output_dir, dataset_path, dataset_name, seed, split_column,
                         split_counts, retrieval_num, scalar_features, list_features,
                         extra_list_columns=None, exclude_group_column=None):
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": dataset_name,
        "dataset_path": str(dataset_path),
        "split_method": f"column:{split_column}" if split_column else "random_80_10_10",
        "seed": int(seed),
        "split_counts": split_counts,
        "retrieval_num": int(retrieval_num),
        "scalar_features": list(scalar_features),
        "list_features": list(list_features),
        "extra_list_columns": list(extra_list_columns or []),
        "exclude_group_column": exclude_group_column,
        "retrieval_pool": "train",
    }
    output_path = Path(output_dir) / "split_manifest.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def create_retrieval_pool(train_path, valid_path, retrieval_pool_path):
    retrieval_pool = pd.read_pickle(train_path).copy()
    retrieval_pool.reset_index(drop=True, inplace=True)
    retrieval_pool.to_pickle(retrieval_pool_path)
    return retrieval_pool


def add_retrieved_label_alias(split_paths):
    for split_path in split_paths:
        df_split = pd.read_pickle(split_path)
        if "retrieved_label_list" not in df_split.columns and "retrieved_label" in df_split.columns:
            df_split["retrieved_label_list"] = df_split["retrieved_label"]
        df_split.to_pickle(split_path)


def _is_missing(value):
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def scalar_key(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(scalar_key(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(scalar_key(item) for item in value))
    if _is_missing(value):
        return ""
    return value


def tokens(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return [scalar_key(item) for item in value if not _is_missing(item)]
    if _is_missing(value):
        return []
    return [scalar_key(value)]


def dedupe_list_columns(path, columns, dataset_name="dataset", required=True):
    if not columns:
        return pd.read_pickle(path)

    data = pd.read_pickle(path)
    for column in columns:
        if column not in data.columns:
            if required:
                raise KeyError(f"{dataset_name} dataset is missing {column}; run 2_preprocess.py first.")
            continue
        data[column] = data[column].apply(lambda value: list(dict.fromkeys(tokens(value))))
    data.to_pickle(path)
    return data


def _feature_weight(pool_size, match_count, weight_mode="idf", absolute_weight=False):
    if weight_mode == "idf":
        weight = math.log((pool_size - match_count + 0.5) / (match_count + 0.5))
    elif weight_mode == "pool_ratio":
        weight = math.log((pool_size + 0.5) / (match_count + 0.5))
    else:
        raise ValueError(f"Unknown retrieval weight mode: {weight_mode}")
    return abs(weight) if absolute_weight else weight


def _build_indexes(retrieval_pool, scalar_features, list_features, weight_mode="idf", absolute_weight=False):
    pool_size = len(retrieval_pool)
    scalar_indexes = {}
    scalar_weights = {}
    list_indexes = {}

    for feature in scalar_features:
        index = defaultdict(list)
        for row_index, value in enumerate(retrieval_pool[feature].tolist()):
            index[scalar_key(value)].append(row_index)
        scalar_indexes[feature] = dict(index)
        scalar_weights[feature] = {
            key: _feature_weight(pool_size, len(indices), weight_mode, absolute_weight)
            for key, indices in index.items()
        }

    for feature in list_features:
        index = defaultdict(list)
        seen_per_token = defaultdict(set)
        for row_index, value in enumerate(retrieval_pool[feature].tolist()):
            for token in tokens(value):
                if row_index not in seen_per_token[token]:
                    index[token].append(row_index)
                    seen_per_token[token].add(row_index)
        list_indexes[feature] = dict(index)

    return scalar_indexes, scalar_weights, list_indexes


def _score_query(query_row, scalar_features, list_features, scalar_indexes, scalar_weights,
                 list_indexes, pool_size, weight_mode="idf", absolute_weight=False):
    scores = defaultdict(float)

    for feature in scalar_features:
        key = scalar_key(query_row[feature])
        postings = scalar_indexes[feature].get(key, [])
        if not postings:
            continue
        weight = scalar_weights[feature][key]
        for row_index in postings:
            scores[row_index] += weight

    for feature in list_features:
        candidate_indices = set()
        for token in tokens(query_row[feature]):
            candidate_indices.update(list_indexes[feature].get(token, []))
        if not candidate_indices:
            continue
        weight = _feature_weight(pool_size, len(candidate_indices), weight_mode, absolute_weight)
        for row_index in candidate_indices:
            scores[row_index] += weight

    return scores


def _sort_items(items, tie_break="asc"):
    tie_multiplier = -1 if tie_break == "desc" else 1
    return sorted(items, key=lambda item: (-item[1], tie_multiplier * item[0]))


def _ordered_indices(indices, tie_break="asc"):
    return sorted(indices, reverse=(tie_break == "desc"))


def _fallback_indices(pool_size, excluded_indices, used, needed, tie_break="asc"):
    if needed <= 0:
        return []

    unused_ordered = _ordered_indices([
        row_index for row_index in range(pool_size)
        if row_index not in excluded_indices and row_index not in used
    ], tie_break)
    all_ordered = _ordered_indices([
        row_index for row_index in range(pool_size)
        if row_index not in excluded_indices
    ], tie_break)
    if not all_ordered:
        return []

    fallback = []
    for ordered in [unused_ordered, all_ordered]:
        while ordered and len(fallback) < needed:
            take = min(needed - len(fallback), len(ordered))
            fallback.extend(ordered[:take])
    return fallback


def _select_top(scores, retrieval_num, pool_size, excluded_indices, tie_break="asc",
                include_zero_score_candidates=False):
    for excluded_index in excluded_indices:
        scores.pop(excluded_index, None)

    if include_zero_score_candidates:
        positive_items = [(index, score) for index, score in scores.items() if score > 0]
        negative_items = [(index, score) for index, score in scores.items() if score < 0]

        scored_items = []
        scored_items.extend(_sort_items(positive_items, tie_break))
        used = {index for index, _ in scored_items}
        if len(scored_items) < retrieval_num:
            index_iter = range(pool_size - 1, -1, -1) if tie_break == "desc" else range(pool_size)
            for index in index_iter:
                if index in excluded_indices or index in used:
                    continue
                if scores.get(index, 0.0) == 0:
                    scored_items.append((index, 0.0))
                    used.add(index)
                    if len(scored_items) >= retrieval_num:
                        break
        if len(scored_items) < retrieval_num:
            scored_items.extend(_sort_items(negative_items, tie_break))
        scored_items = scored_items[:retrieval_num]
    else:
        tie_multiplier = -1 if tie_break == "desc" else 1
        scored_items = heapq.nsmallest(
            retrieval_num,
            scores.items(),
            key=lambda item: (-item[1], tie_multiplier * item[0]),
        )

    selected_indices = [row_index for row_index, _ in scored_items]
    selected_scores = [float(score) for _, score in scored_items]

    if len(selected_indices) < retrieval_num:
        used = set(selected_indices)
        fallback = _fallback_indices(
            pool_size, excluded_indices, used, retrieval_num - len(selected_indices), tie_break
        )
        selected_indices.extend(fallback)
        selected_scores.extend([0.0] * len(fallback))

    return selected_indices[:retrieval_num], selected_scores[:retrieval_num]


def retrieval_data(retrieval_num, data_path, retrieval_pool_path, scalar_features, list_features,
                   dataset_name="dataset", weight_mode="idf", absolute_weight=False,
                   tie_break="asc", include_zero_score_candidates=False,
                   exclude_group_column=None):
    retrieval_pool = pd.read_pickle(retrieval_pool_path)
    data = pd.read_pickle(data_path)
    required = set(scalar_features + list_features + ["image_id", "label"])
    if exclude_group_column:
        required.add(exclude_group_column)
    missing = required - set(data.columns) | required - set(retrieval_pool.columns)
    if missing:
        raise KeyError(f"{dataset_name} retrieval input is missing required columns: {sorted(missing)}")

    scalar_indexes, scalar_weights, list_indexes = _build_indexes(
        retrieval_pool, scalar_features, list_features, weight_mode, absolute_weight
    )
    pool_size = len(retrieval_pool)
    pool_ids = retrieval_pool["image_id"].tolist()
    pool_id_keys = [str(image_id) for image_id in pool_ids]
    pool_positions = defaultdict(list)
    for index, image_id in enumerate(pool_id_keys):
        pool_positions[image_id].append(index)
    group_positions = defaultdict(list)
    if exclude_group_column:
        for index, group in enumerate(retrieval_pool[exclude_group_column].astype(str).tolist()):
            group_positions[group].append(index)
    pool_labels = retrieval_pool["label"].tolist()

    retrieved_item_id_list = []
    retrieved_item_similarity_list = []
    retrieved_label_list = []

    for _, query_row in tqdm(data.iterrows(), total=len(data)):
        query_id = str(query_row["image_id"])
        excluded_indices = set(pool_positions.get(query_id, []))
        if exclude_group_column:
            excluded_indices.update(group_positions.get(str(query_row[exclude_group_column]), []))
        scores = _score_query(
            query_row,
            scalar_features,
            list_features,
            scalar_indexes,
            scalar_weights,
            list_indexes,
            pool_size,
            weight_mode,
            absolute_weight,
        )
        selected_indices, selected_scores = _select_top(
            scores, retrieval_num, pool_size, excluded_indices, tie_break, include_zero_score_candidates
        )

        retrieved_item_id_list.append([pool_ids[index] for index in selected_indices])
        retrieved_item_similarity_list.append(selected_scores)
        retrieved_label_list.append([pool_labels[index] for index in selected_indices])

    data["retrieved_item_id"] = retrieved_item_id_list
    data["retrieved_item_similarity"] = retrieved_item_similarity_list
    data["retrieved_label"] = retrieved_label_list
    data.to_pickle(data_path)


def run_retrieval_pipeline(dataset_path, output_dir, retrieval_num, scalar_features, list_features,
                           extra_list_columns=None, seed=DEFAULT_SEED, dataset_name="dataset",
                           weight_mode="idf", absolute_weight=False, tie_break="asc",
                           include_zero_score_candidates=False, split_column=None,
                           exclude_group_column=None):
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir) if output_dir else dataset_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.pkl"
    valid_path = output_dir / "valid.pkl"
    test_path = output_dir / "test.pkl"
    retrieval_pool_path = output_dir / "retrieval_pool.pkl"

    validate_retrieval_features(scalar_features + list_features + (extra_list_columns or []))
    dedupe_columns = list(dict.fromkeys(list_features + (extra_list_columns or [])))
    dedupe_list_columns(dataset_path, dedupe_columns, dataset_name)
    if split_column:
        split_counts = split_by_column_and_save_pkl(dataset_path, train_path, valid_path, test_path, split_column)
    else:
        split_counts = split_and_save_pkl(dataset_path, train_path, valid_path, test_path, seed)
    print("Split dataset done!")
    write_split_manifest(
        output_dir, dataset_path, dataset_name, seed, split_column, split_counts,
        retrieval_num, scalar_features, list_features, extra_list_columns, exclude_group_column
    )

    create_retrieval_pool(train_path, valid_path, retrieval_pool_path)
    print("Create retrieval pool done!")

    retrieval_data(
        retrieval_num, train_path, retrieval_pool_path, scalar_features, list_features,
        dataset_name, weight_mode, absolute_weight, tie_break, include_zero_score_candidates,
        exclude_group_column
    )
    retrieval_data(
        retrieval_num, valid_path, retrieval_pool_path, scalar_features, list_features,
        dataset_name, weight_mode, absolute_weight, tie_break, include_zero_score_candidates,
        exclude_group_column
    )
    retrieval_data(
        retrieval_num, test_path, retrieval_pool_path, scalar_features, list_features,
        dataset_name, weight_mode, absolute_weight, tie_break, include_zero_score_candidates,
        exclude_group_column
    )
    print("Retrieval done!")

    add_retrieved_label_alias([train_path, valid_path, test_path])
    print("Stack retrieved feature done!")
