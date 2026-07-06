import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval_utils import run_retrieval_pipeline


SCALAR_FEATURES = [
    "user_id",
    "date_posted",
    "date_taken",
    "size",
    "num_sets",
    "num_groups",
    "avg_group_members",
    "avg_group_photos",
]
LIST_FEATURES = ["tags", "nouns", "verbs"]
EXTRA_LIST_COLUMNS = ["adjectives"]


def parse_args():
    parser = argparse.ArgumentParser(description="Split ICIP and retrieve top-K train-pool neighbors.")
    parser.add_argument("--dataset_path", default="datasets/ICIP/dataset.pkl", help="Input ICIP dataset pickle")
    parser.add_argument("--output_dir", default=None, help="Output directory for train/valid/test/retrieval_pool")
    parser.add_argument("--retrieval_num", default=50, type=int, help="Number of retrieved UGCs per query")
    parser.add_argument("--seed", default=12, type=int, help="Random seed used only when --split_column is empty")
    parser.add_argument(
        "--split_column",
        default="split",
        help="Column containing temporal train/valid/test splits. Use empty string to request a random split.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    run_retrieval_pipeline(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        retrieval_num=args.retrieval_num,
        scalar_features=SCALAR_FEATURES,
        list_features=LIST_FEATURES,
        extra_list_columns=EXTRA_LIST_COLUMNS,
        seed=args.seed,
        dataset_name="ICIP",
        weight_mode="idf",
        absolute_weight=True,
        tie_break="desc",
        include_zero_score_candidates=True,
        split_column=args.split_column or None,
    )
    print(f"Runtime: {(time.time() - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
