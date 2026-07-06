import argparse
import ast
import json
import math
import random
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval_utils import SPLIT_VALUES, assign_temporal_splits


DEFAULT_RAW_DIR = Path("datasets/raw_dataset/Instagram")
LEGACY_RAW_DIRS = (
    Path("datasets/raw_dataset/INSTAGRAM"),
    Path("datasets/origin_dataset/INS"),
)
MAPPING_FILE_NAME = "JSON-Image_files_mapping.txt"
METADATA_ARCHIVE = Path("Post_metadata") / "posts_info.zip"


class MetadataReadError(RuntimeError):
    pass


@dataclass(frozen=True)
class RawPostRecord:
    user_name: str
    json_name: str
    post_id: str
    image_names: tuple

    @property
    def image_id(self):
        return f"{self.user_name}-{self.post_id}"

    @property
    def metadata_rel_path(self):
        return f"info/{self.user_name}-{self.json_name}"

    @property
    def image_rel_paths(self):
        return [f"image/{self.user_name}-{image_name}" for image_name in self.image_names]


def _candidate_raw_dirs(raw_dir):
    raw_dir = Path(raw_dir)
    candidates = [raw_dir]
    candidates.extend(path for path in LEGACY_RAW_DIRS if path != raw_dir)
    return candidates


def _find_file(raw_dir, file_name, explicit_path=None):
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path

    for directory in _candidate_raw_dirs(raw_dir):
        path = directory / file_name
        if path.exists():
            return path
    searched = ", ".join(str(path / file_name) for path in _candidate_raw_dirs(raw_dir))
    raise FileNotFoundError(f"Could not find {file_name}. Searched: {searched}")


def _find_optional_file(raw_dir, file_name, explicit_path=None):
    if explicit_path:
        return _find_file(raw_dir, file_name, explicit_path)

    for directory in _candidate_raw_dirs(raw_dir):
        path = directory / file_name
        if path.exists():
            return path
    return None


def _safe_image_list(value):
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, (list, tuple)):
        return []
    return tuple(str(item).strip() for item in parsed if str(item).strip())


def iter_mapping_records(mapping_path):
    with open(mapping_path, encoding="utf-8", errors="replace") as handle:
        next(handle, None)
        for line_number, line in enumerate(handle, 2):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                raise ValueError(f"Bad mapping line {line_number}: {line[:120]!r}")
            user_name, json_name, image_text = parts[0].strip(), parts[1].strip(), parts[2].strip()
            image_names = _safe_image_list(image_text)
            if not user_name or not json_name or not image_names:
                continue
            yield RawPostRecord(
                user_name=user_name,
                json_name=json_name,
                post_id=Path(json_name).stem,
                image_names=image_names,
            )


def select_records(mapping_path, sample_size=None, seed=12, excluded_keys=None):
    excluded_keys = set(excluded_keys or [])
    if sample_size is None:
        return [
            record for record in iter_mapping_records(mapping_path)
            if record.image_id not in excluded_keys
        ]

    sample_size = int(sample_size)
    if sample_size <= 0:
        raise ValueError("--sample_size must be positive")

    rng = random.Random(seed)
    reservoir = []
    seen = 0
    for record in iter_mapping_records(mapping_path):
        if record.image_id in excluded_keys:
            continue
        seen += 1
        if len(reservoir) < sample_size:
            reservoir.append(record)
            continue
        replacement_index = rng.randrange(seen)
        if replacement_index < sample_size:
            reservoir[replacement_index] = record

    if len(reservoir) < sample_size:
        print(f"Requested {sample_size} posts, but only {len(reservoir)} valid mapping rows were found.")
    return reservoir


def _metadata_candidates(root, record):
    root = Path(root)
    name = f"{record.user_name}-{record.json_name}"
    candidates = [
        root / record.metadata_rel_path,
        root / name,
        root / "info" / name,
        root / record.json_name,
        root / "info" / record.json_name,
    ]
    return list(dict.fromkeys(candidates))


def _metadata_exists(metadata_roots, record):
    for root in metadata_roots:
        if root is None:
            continue
        for path in _metadata_candidates(root, record):
            if path.exists():
                return True
    return False


def _read_metadata_json(metadata_roots, record):
    for root in metadata_roots:
        if root is None:
            continue
        for path in _metadata_candidates(root, record):
            if path.exists():
                try:
                    with open(path, encoding="utf-8", errors="replace") as handle:
                        metadata = json.load(handle)
                except JSONDecodeError as exc:
                    raise MetadataReadError(f"Invalid metadata JSON for {record.metadata_rel_path}: {path}") from exc
                if not isinstance(metadata, dict):
                    raise MetadataReadError(f"Metadata JSON is not an object for {record.metadata_rel_path}: {path}")
                return metadata
    raise FileNotFoundError(f"Metadata JSON not found for {record.metadata_rel_path}")


def _find_7z(explicit_path=None):
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    for name in ["7z", "7zz", "7za"]:
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    candidates.extend([
        Path(r"C:\Program Files\7-Zip\7z.exe"),
        Path(r"C:\Program Files (x86)\7-Zip\7z.exe"),
    ])
    for path in candidates:
        if path and path.exists():
            return path
    return None


def _extract_missing_metadata(records, metadata_archive, metadata_cache_dir, seven_zip=None, existing_roots=None):
    metadata_archive = Path(metadata_archive) if metadata_archive else None
    metadata_cache_dir = Path(metadata_cache_dir) if metadata_cache_dir else None
    if metadata_archive is None or metadata_cache_dir is None:
        return
    if not metadata_archive.exists():
        raise FileNotFoundError(f"Metadata archive not found: {metadata_archive}")

    search_roots = [metadata_cache_dir, *(existing_roots or [])]
    missing = [record.metadata_rel_path for record in records if not _metadata_exists(search_roots, record)]
    if not missing:
        return

    seven_zip_path = _find_7z(seven_zip)
    if seven_zip_path is None:
        raise FileNotFoundError(
            "7-Zip was not found. Extract the Instagram metadata archive manually "
            "or pass --seven_zip with the path to 7z.exe."
        )

    metadata_cache_dir.mkdir(parents=True, exist_ok=True)
    list_path = metadata_cache_dir / "metadata_members.txt"
    with open(list_path, "w", encoding="utf-8") as handle:
        for member in missing:
            handle.write(member.replace("/", "\\") + "\n")

    command = [
        str(seven_zip_path),
        "x",
        str(metadata_archive),
        f"@{list_path}",
        f"-o{metadata_cache_dir}",
        "-y",
    ]
    print(f"Extracting {len(missing)} metadata JSON files to {metadata_cache_dir}")
    subprocess.run(command, check=True)


def _caption_from_json(metadata):
    edges = metadata.get("edge_media_to_caption", {}).get("edges", [])
    if not edges:
        return ""
    node = edges[0].get("node", {}) if isinstance(edges[0], dict) else {}
    return str(node.get("text", "") or "")


def _safe_count(metadata, path):
    current = metadata
    for key in path:
        if not isinstance(current, dict):
            return 0
        current = current.get(key)
    try:
        return int(current)
    except (TypeError, ValueError):
        return 0


def _row_from_raw_metadata(record, metadata):
    like_count = _safe_count(metadata, ["edge_media_preview_like", "count"])
    owner = metadata.get("owner", {}) if isinstance(metadata.get("owner"), dict) else {}
    timestamp = _safe_count(metadata, ["taken_at_timestamp"])
    image_rel_paths = record.image_rel_paths
    return {
        "image_id": record.image_id,
        "post_id": record.post_id,
        "source_unit": "post",
        "metadata_rel_path": record.metadata_rel_path,
        "image_rel_path": image_rel_paths[0] if image_rel_paths else "",
        "image_rel_paths": image_rel_paths,
        "image_count": len(image_rel_paths),
        "text": _caption_from_json(metadata),
        "label": math.log2(like_count + 1),
        "like_count": like_count,
        "user_id": str(owner.get("id", "") or ""),
        "user_name": str(owner.get("username", record.user_name) or record.user_name),
        "taken_timestamp": timestamp,
        "is_video": bool(metadata.get("is_video", False)),
    }


def _print_split_summary(dataframe):
    print(f"Rows: {len(dataframe)}, unique image_id: {dataframe['image_id'].nunique()}")
    if "split" in dataframe.columns:
        print("Split sizes:")
        print(dataframe["split"].value_counts().reindex(SPLIT_VALUES).fillna(0).astype(int).to_string())
    print(
        "Label min/mean/max: "
        f"{dataframe['label'].min():.4f} / {dataframe['label'].mean():.4f} / {dataframe['label'].max():.4f}"
    )
    if "taken_timestamp" in dataframe.columns:
        for split_name in SPLIT_VALUES:
            subset = dataframe[dataframe["split"] == split_name]
            if len(subset) == 0:
                continue
            print(
                f"{split_name} timestamp range: "
                f"{int(subset['taken_timestamp'].min())} - {int(subset['taken_timestamp'].max())}"
            )


def _read_rows_from_records(records, metadata_roots):
    rows = []
    skipped = Counter()
    examples = {}
    for record in tqdm(records, desc="Reading Instagram metadata"):
        try:
            metadata = _read_metadata_json(metadata_roots, record)
        except FileNotFoundError as exc:
            skipped["missing_metadata"] += 1
            examples.setdefault("missing_metadata", str(exc))
            continue
        except MetadataReadError as exc:
            skipped["invalid_metadata"] += 1
            examples.setdefault("invalid_metadata", str(exc))
            continue
        row = _row_from_raw_metadata(record, metadata)
        if int(row["taken_timestamp"]) <= 0:
            skipped["invalid_timestamp"] += 1
            examples.setdefault("invalid_timestamp", f"{record.metadata_rel_path} has taken_timestamp={row['taken_timestamp']}")
            continue
        rows.append(row)
    return rows, skipped, examples


def _print_skip_summary(skipped, examples):
    if not skipped:
        return
    summary = ", ".join(f"{key}={value}" for key, value in sorted(skipped.items()))
    print(f"Skipped metadata records: {summary}")
    for key in sorted(examples):
        print(f"Example {key}: {examples[key]}")


def build_best_practice_dataset(
    raw_dir=DEFAULT_RAW_DIR,
    mapping_path=None,
    metadata_root=None,
    metadata_archive=None,
    metadata_cache_dir=None,
    output_path="datasets/INS/dataset.pkl",
    sample_size=300000,
    seed=12,
    mode="default",
    seven_zip=None,
):
    raw_dir = Path(raw_dir)
    mapping_path = _find_file(raw_dir, MAPPING_FILE_NAME, mapping_path)
    metadata_archive = _find_optional_file(raw_dir, METADATA_ARCHIVE, metadata_archive)
    if metadata_archive is None and metadata_root is None:
        raise FileNotFoundError(
            "Instagram metadata was not found. Provide --metadata_root with extracted JSON files "
            "or place Post_metadata/posts_info.zip under --raw_dir."
        )
    metadata_cache_dir = Path(metadata_cache_dir) if metadata_cache_dir else Path(output_path).parent / "metadata_cache"
    metadata_roots = [Path(metadata_root) if metadata_root else None, metadata_cache_dir]

    target_size = None if mode == "full" else int(sample_size)
    selected_records = select_records(mapping_path, target_size, seed)
    print(f"Selected {len(selected_records)} raw post records from {mapping_path}")

    rows = []
    attempted_keys = set()
    skipped_total = Counter()
    examples = {}
    round_records = selected_records
    round_index = 0
    dataframe = pd.DataFrame()

    while round_records:
        _extract_missing_metadata(
            round_records,
            metadata_archive,
            metadata_cache_dir,
            seven_zip,
            [Path(metadata_root)] if metadata_root else [],
        )

        new_rows, skipped, new_examples = _read_rows_from_records(round_records, metadata_roots)
        rows.extend(new_rows)
        skipped_total.update(skipped)
        for key, value in new_examples.items():
            examples.setdefault(key, value)
        attempted_keys.update(record.image_id for record in round_records)

        dataframe = pd.DataFrame(rows).drop_duplicates("image_id", keep="first")
        if target_size is None or len(dataframe) >= target_size:
            break

        needed = target_size - len(dataframe)
        round_index += 1
        replacement_count = max(needed * 2, min(target_size, 5000))
        print(
            f"Collected {len(dataframe)}/{target_size} valid posts; "
            f"sampling {replacement_count} replacement candidates."
        )
        round_records = select_records(
            mapping_path,
            replacement_count,
            seed + round_index,
            excluded_keys=attempted_keys,
        )

    if not rows:
        raise RuntimeError("No Instagram metadata rows were built.")
    _print_skip_summary(skipped_total, examples)

    if target_size is not None:
        if len(dataframe) < target_size:
            print(f"Warning: requested {target_size} valid posts, but only built {len(dataframe)}.")
        elif len(dataframe) > target_size:
            dataframe = dataframe.iloc[:target_size].copy()
    dataframe = assign_temporal_splits(dataframe, "taken_timestamp", tie_break_columns=["image_id"])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_pickle(output_path)
    _print_split_summary(dataframe)
    print(f"Saved Instagram dataset to {output_path}")
    return dataframe


def parse_args():
    parser = argparse.ArgumentParser(description="Build Instagram dataset files for SKAPP.")
    parser.add_argument(
        "--mode",
        default="default",
        choices=["default", "full"],
        help="default: 300k post-level temporal split; full: all post-level rows",
    )
    parser.add_argument("--raw_dir", default=str(DEFAULT_RAW_DIR), help="Official Instagram raw dataset directory")
    parser.add_argument("--mapping_path", default=None, help="Optional JSON-image mapping file path")
    parser.add_argument("--metadata_root", default=None, help="Directory containing extracted Instagram JSON metadata")
    parser.add_argument("--metadata_archive", default=None, help="Optional posts_info.zip path")
    parser.add_argument("--metadata_cache_dir", default=None, help="Cache directory for selected metadata JSON files")
    parser.add_argument("--output_path", default="datasets/INS/dataset.pkl", help="Output dataset pickle path")
    parser.add_argument("--sample_size", default=300000, type=int, help="Default-mode sample size")
    parser.add_argument("--seed", default=12, type=int, help="Sampling seed for default mode")
    parser.add_argument("--seven_zip", default=None, help="Optional path to 7z executable for multivolume archives")
    return parser.parse_args()


def main():
    args = parse_args()
    build_best_practice_dataset(
        raw_dir=args.raw_dir,
        mapping_path=args.mapping_path,
        metadata_root=args.metadata_root,
        metadata_archive=args.metadata_archive,
        metadata_cache_dir=args.metadata_cache_dir,
        output_path=args.output_path,
        sample_size=args.sample_size,
        seed=args.seed,
        mode=args.mode,
        seven_zip=args.seven_zip,
    )


if __name__ == "__main__":
    main()
