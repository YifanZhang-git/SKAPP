import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval_utils import run_retrieval_pipeline


DEFAULT_SCALAR_FEATURES = ["user_id"]
LIST_FEATURES = []


def _parse_feature_list(value):
    if value is None:
        return None
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Split Instagram and retrieve top-K train-pool neighbors.")
    parser.add_argument("--dataset_path", default="datasets/INS/dataset.pkl", help="Input Instagram dataset pickle")
    parser.add_argument("--output_dir", default=None, help="Output directory for train/valid/test/retrieval_pool")
    parser.add_argument("--retrieval_num", default=50, type=int, help="Number of retrieved UGCs per query")
    parser.add_argument("--seed", default=12, type=int, help="Random seed used only when --split_column is empty")
    parser.add_argument(
        "--scalar_features",
        default=None,
        type=_parse_feature_list,
        help="Comma-separated scalar retrieval features. Defaults to user_id only.",
    )
    parser.add_argument(
        "--split_column",
        default="split",
        help="Column containing temporal train/valid/test splits. Use empty string to request a random split.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    dataset = pd.read_pickle(args.dataset_path)
    scalar_features = args.scalar_features or DEFAULT_SCALAR_FEATURES

    dataset_columns = set(dataset.columns)
    split_column = args.split_column or None
    exclude_group_column = "post_id" if "post_id" in dataset_columns else None
    del dataset

    print(f"Scalar retrieval features: {scalar_features}")
    print(f"Split column: {split_column}")
    print(f"Exclude group column: {exclude_group_column}")

    run_retrieval_pipeline(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        retrieval_num=args.retrieval_num,
        scalar_features=scalar_features,
        list_features=LIST_FEATURES,
        seed=args.seed,
        dataset_name="INS",
        weight_mode="idf",
        tie_break="desc",
        include_zero_score_candidates=True,
        split_column=split_column,
        exclude_group_column=exclude_group_column,
    )
    print(f"Runtime: {(time.time() - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
