import argparse
import ast
import io
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
import torch
from angle_emb import AnglE
from PIL import Image
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor, ViTImageProcessor, ViTModel


DEFAULT_RAW_DIR = Path("datasets/raw_dataset/Instagram")
LEGACY_RAW_DIRS = (
    Path("datasets/raw_dataset/INSTAGRAM"),
    Path("datasets/origin_dataset/INS"),
)
IMAGE_ARCHIVE = Path("Post_images") / "posts_image.zip"


def _batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield start, items[start:start + batch_size]


def _batch_count(items, batch_size):
    return (len(items) + batch_size - 1) // batch_size


def _column_ready(dataframe, columns):
    return all(column in dataframe.columns and len(dataframe[column]) == len(dataframe) for column in columns)


def _candidate_raw_dirs():
    return [DEFAULT_RAW_DIR, *LEGACY_RAW_DIRS]


def _find_image_zip(image_zip=None):
    if image_zip:
        path = Path(image_zip)
        if not path.exists():
            raise FileNotFoundError(f"Image zip not found: {path}")
        return path

    for directory in _candidate_raw_dirs():
        path = directory / IMAGE_ARCHIVE
        if path.exists():
            return path
    return None


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


def _find_image_root(image_root=None):
    if image_root:
        path = Path(image_root)
        if not path.exists():
            raise FileNotFoundError(f"Image root not found: {path}")
        return path

    for path in [Path("datasets/INS/pic"), DEFAULT_RAW_DIR, *LEGACY_RAW_DIRS]:
        if path.exists() and ((path / "image").exists() or any(path.glob("*.jpg"))):
            return path
    return None


def _as_list(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, (list, tuple)):
                    return [str(item) for item in parsed]
            except (SyntaxError, ValueError):
                pass
        return [stripped]
    return [str(value)]


def _image_records(meta_data):
    if "image_rel_paths" in meta_data.columns:
        rel_paths = meta_data["image_rel_paths"].apply(_as_list).tolist()
    elif "image_rel_path" in meta_data.columns:
        rel_paths = meta_data["image_rel_path"].apply(lambda value: [str(value)] if str(value) else []).tolist()
    else:
        rel_paths = [[f"{image_id}.jpg"] for image_id in meta_data["image_id"].astype(str)]
    return list(zip(meta_data["image_id"].astype(str).tolist(), rel_paths))


def _limit_paths(paths, max_images):
    paths = [str(path).replace("\\", "/") for path in paths if str(path).strip()]
    if max_images is None or int(max_images) <= 0:
        return paths
    return paths[:int(max_images)]


def _combined_image_limit(*limits):
    normalized = [int(limit) for limit in limits if limit is not None]
    if not normalized:
        return 0
    if any(limit <= 0 for limit in normalized):
        return 0
    return max(normalized)


def _cache_candidates(cache_dir, rel_path):
    rel_path = Path(str(rel_path).replace("\\", os.sep))
    return [
        Path(cache_dir) / rel_path,
        Path(cache_dir) / rel_path.name,
        Path(cache_dir) / "image" / rel_path.name,
    ]


def _cached(cache_dir, rel_path):
    return any(path.exists() for path in _cache_candidates(cache_dir, rel_path))


def prepare_image_cache(dataset_path, image_archive=None, image_cache_dir=None,
                        max_images_per_post=0, seven_zip=None):
    image_archive = _find_image_zip(image_archive)
    if image_archive is None:
        raise FileNotFoundError("Instagram image archive was not found")

    image_cache_dir = Path(image_cache_dir) if image_cache_dir else Path(dataset_path).parent / "image_cache"
    image_cache_dir.mkdir(parents=True, exist_ok=True)

    meta_data = pd.read_pickle(dataset_path)
    members = []
    seen = set()
    for _, rel_paths in _image_records(meta_data):
        for rel_path in _limit_paths(rel_paths, max_images_per_post):
            rel_path = str(rel_path).replace("\\", "/")
            if rel_path in seen or _cached(image_cache_dir, rel_path):
                continue
            seen.add(rel_path)
            members.append(rel_path)

    if not members:
        print(f"Image cache is ready: {image_cache_dir}")
        return image_cache_dir

    seven_zip_path = _find_7z(seven_zip)
    if seven_zip_path is None:
        raise FileNotFoundError(
            "7-Zip was not found. Extract Instagram images manually or pass --seven_zip with the path to 7z.exe."
        )

    list_path = image_cache_dir / "image_members.txt"
    with open(list_path, "w", encoding="utf-8") as handle:
        for member in members:
            handle.write(member.replace("/", "\\") + "\n")

    command = [
        str(seven_zip_path),
        "x",
        str(image_archive),
        f"@{list_path}",
        f"-o{image_cache_dir}",
        "-y",
    ]
    print(f"Extracting {len(members)} Instagram images to {image_cache_dir}")
    subprocess.run(command, check=True)
    return image_cache_dir


class ImageStore:
    def __init__(self, image_root=None, image_zip=None):
        self.image_zip = _find_image_zip(image_zip) if image_zip or image_root is None else None
        self.image_root = None if self.image_zip else _find_image_root(image_root)
        self.zip_file = None

        if self.image_zip is None and self.image_root is None:
            raise FileNotFoundError("No Instagram image zip or image root was found")

    def __enter__(self):
        if self.image_zip:
            try:
                self.zip_file = zipfile.ZipFile(self.image_zip)
            except zipfile.BadZipFile as exc:
                raise RuntimeError(
                    "Python cannot read the multivolume Instagram image archive directly. "
                    "Extract selected images with 7-Zip and pass --image_root."
                ) from exc
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.zip_file:
            self.zip_file.close()

    def _zip_candidates(self, rel_path, image_id):
        rel_path = str(rel_path).replace("\\", "/")
        name = Path(rel_path).name
        candidates = [rel_path, name, f"image/{name}"]
        if image_id:
            candidates.extend([f"{image_id}.jpg", f"image/{image_id}.jpg"])
        return list(dict.fromkeys(candidates))

    def _root_candidates(self, rel_path, image_id):
        rel_path = Path(str(rel_path).replace("\\", os.sep))
        name = rel_path.name
        candidates = [self.image_root / rel_path, self.image_root / name, self.image_root / "image" / name]
        if image_id:
            candidates.extend([self.image_root / f"{image_id}.jpg", self.image_root / "image" / f"{image_id}.jpg"])
        return list(dict.fromkeys(candidates))

    def open(self, rel_path, image_id=None):
        try:
            if self.zip_file:
                for candidate in self._zip_candidates(rel_path, image_id):
                    try:
                        with self.zip_file.open(candidate) as handle:
                            return Image.open(io.BytesIO(handle.read())).convert("RGB")
                    except KeyError:
                        continue
                return None

            for candidate in self._root_candidates(rel_path, image_id):
                if candidate.exists():
                    with Image.open(candidate) as image:
                        return image.convert("RGB")
            return None
        except Exception:
            return None


def _zero_embedding():
    return [[0.0] * 768]


def image2text(meta_file_path, image_root=None, image_zip=None, batch_size=16, device=None,
               max_images_per_post=1, force=False):
    meta_data = pd.read_pickle(meta_file_path)
    if not force and _column_ready(meta_data, ["image_to_text"]):
        print("image_to_text already exists; skipping image captioning.")
        return 0

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", use_fast=True)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
    model.eval()

    records = _image_records(meta_data)
    image_to_text_list = []
    missing_or_broken = 0

    with ImageStore(image_root, image_zip) as image_store:
        for _, batch_records in tqdm(_batched(records, batch_size), total=_batch_count(records, batch_size)):
            captions = ["0"] * len(batch_records)
            images = []
            owners = []

            for record_pos, (image_id, rel_paths) in enumerate(batch_records):
                selected_paths = _limit_paths(rel_paths, max_images_per_post)
                for rel_path in selected_paths:
                    image = image_store.open(rel_path, image_id)
                    if image is None:
                        missing_or_broken += 1
                        continue
                    images.append(image)
                    owners.append(record_pos)

            if images:
                inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
                with torch.inference_mode():
                    generated_ids = model.generate(**inputs)
                decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)

                grouped = [[] for _ in batch_records]
                for record_pos, text in zip(owners, decoded):
                    grouped[record_pos].append(text)
                captions = [" ".join(parts) if parts else "0" for parts in grouped]

            image_to_text_list.extend(captions)

    meta_data["image_to_text"] = image_to_text_list
    meta_data.to_pickle(meta_file_path)
    print(f"image2text done; missing or unreadable images: {missing_or_broken}")
    return 0


def image2vec(meta_file_path, image_root=None, image_zip=None, batch_size=64, device=None,
              max_images_per_post=0, force=False):
    meta_data = pd.read_pickle(meta_file_path)
    if not force and _column_ready(meta_data, ["mean_pooling_vec", "cls_vec"]):
        print("Image embeddings already exist; skipping image2vec.")
        return 0

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
    model.eval()

    records = _image_records(meta_data)
    mean_pooling_vec_list = []
    cls_vec_list = []
    missing_or_broken = 0

    with ImageStore(image_root, image_zip) as image_store:
        for _, batch_records in tqdm(_batched(records, batch_size), total=_batch_count(records, batch_size)):
            grouped_mean = [[] for _ in batch_records]
            grouped_cls = [[] for _ in batch_records]
            images = []
            owners = []

            for record_pos, (image_id, rel_paths) in enumerate(batch_records):
                selected_paths = _limit_paths(rel_paths, max_images_per_post)
                for rel_path in selected_paths:
                    image = image_store.open(rel_path, image_id)
                    if image is None:
                        missing_or_broken += 1
                        continue
                    images.append(image)
                    owners.append(record_pos)

            if images:
                inputs = processor(images=images, return_tensors="pt").to(device)
                with torch.inference_mode():
                    outputs = model(**inputs)

                cls_output = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
                mean_output = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()

                for encoded_pos, record_pos in enumerate(owners):
                    grouped_mean[record_pos].append(mean_output[encoded_pos])
                    grouped_cls[record_pos].append(cls_output[encoded_pos])

            for mean_values, cls_values in zip(grouped_mean, grouped_cls):
                if mean_values:
                    mean_pooling_vec_list.append([np.mean(mean_values, axis=0).astype(np.float32).tolist()])
                    cls_vec_list.append([np.mean(cls_values, axis=0).astype(np.float32).tolist()])
                else:
                    mean_pooling_vec_list.append(_zero_embedding())
                    cls_vec_list.append(_zero_embedding())

    meta_data["mean_pooling_vec"] = mean_pooling_vec_list
    meta_data["cls_vec"] = cls_vec_list
    meta_data.to_pickle(meta_file_path)
    print(f"image2vec done; missing or unreadable images: {missing_or_broken}")
    return 0


def merged_text_create_and_to_vec(meta_file_path, batch_size=256, device=None, force=False):
    meta_data = pd.read_pickle(meta_file_path)
    if not force and _column_ready(meta_data, ["merged_text", "merged_text_vec"]):
        print("Merged text embeddings already exist; skipping text embedding.")
        return 0

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    angel = AnglE.from_pretrained("SeanLee97/angle-bert-base-uncased-nli-en-v1", pooling_strategy="cls_avg")
    if hasattr(angel, "to"):
        angel = angel.to(device)
    elif str(device).startswith("cuda") and hasattr(angel, "cuda"):
        angel = getattr(angel, "cuda")()

    text_list = meta_data["text"].fillna("").astype(str).tolist()
    image_to_text_list = meta_data["image_to_text"].fillna("").astype(str).tolist()
    merged_inputs = [" ".join([text, image_text]).strip() for text, image_text in zip(text_list, image_to_text_list)]
    merged_text_list = []
    merged_text_vec_list = []

    for _, batch_texts in tqdm(_batched(merged_inputs, batch_size), total=_batch_count(merged_inputs, batch_size)):
        merged_text_list.extend(batch_texts)
        text_embedding = angel.encode(batch_texts, to_numpy=True)
        if text_embedding.ndim == 1:
            text_embedding = text_embedding.reshape(1, -1)
        for index in range(text_embedding.shape[0]):
            merged_text_vec_list.append(text_embedding[index:index + 1].tolist())

    meta_data["merged_text"] = merged_text_list
    meta_data["merged_text_vec"] = merged_text_vec_list
    meta_data.to_pickle(meta_file_path)
    return 0


def merged_text2nouns2verb2adj(meta_file_path, batch_size=256, force=False):
    meta_data = pd.read_pickle(meta_file_path)
    if not force and _column_ready(meta_data, ["nouns", "verbs", "adjectives"]):
        print("POS features already exist; skipping spaCy extraction.")
        return 0

    nlp = spacy.load("en_core_web_sm")
    n_nouns = []
    n_verbs = []
    n_adjectives = []
    merged_text_list = meta_data["merged_text"].fillna("").astype(str).tolist()

    for doc in tqdm(nlp.pipe(merged_text_list, batch_size=batch_size), total=len(merged_text_list)):
        n_nouns.append([token.text for token in doc if token.pos_ == "NOUN"])
        n_verbs.append([token.text for token in doc if token.pos_ == "VERB"])
        n_adjectives.append([token.text for token in doc if token.pos_ == "ADJ"])

    meta_data["nouns"] = n_nouns
    meta_data["verbs"] = n_verbs
    meta_data["adjectives"] = n_adjectives
    meta_data.to_pickle(meta_file_path)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Extract Instagram image, text, and POS features.")
    parser.add_argument("--dataset_path", default="datasets/INS/dataset.pkl", help="Instagram dataset pickle path")
    parser.add_argument("--image_root", default=None, help="Directory containing extracted Instagram images")
    parser.add_argument("--image_zip", default=None, help="Standard image zip path; multivolume zips require cache extraction")
    parser.add_argument("--prepare_image_cache", action="store_true", help="Extract selected images from archive before feature extraction")
    parser.add_argument("--image_cache_dir", default=None, help="Directory for selected extracted images")
    parser.add_argument("--seven_zip", default=None, help="Optional path to 7z executable for multivolume archives")
    parser.add_argument("--device", default=None, help="Torch device, for example cuda:0 or cpu")
    parser.add_argument("--caption_batch_size", default=16, type=int, help="BLIP captioning batch size")
    parser.add_argument("--image_batch_size", default=64, type=int, help="ViT image embedding batch size")
    parser.add_argument("--text_batch_size", default=256, type=int, help="AnglE text embedding batch size")
    parser.add_argument("--spacy_batch_size", default=256, type=int, help="spaCy pipe batch size")
    parser.add_argument(
        "--max_caption_images_per_post",
        default=1,
        type=int,
        help="Maximum images captioned per post; 0 means all images",
    )
    parser.add_argument(
        "--max_embedding_images_per_post",
        default=0,
        type=int,
        help="Maximum images embedded per post; 0 means all images",
    )
    parser.add_argument("--force", action="store_true", help="Recompute features even if output columns exist")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    image_root = args.image_root
    image_zip = args.image_zip

    if args.prepare_image_cache:
        cache_image_limit = _combined_image_limit(
            args.max_caption_images_per_post,
            args.max_embedding_images_per_post,
        )
        image_root = prepare_image_cache(
            args.dataset_path,
            args.image_zip,
            args.image_cache_dir,
            cache_image_limit,
            args.seven_zip,
        )
        image_zip = None

    image2text(
        args.dataset_path,
        image_root,
        image_zip,
        args.caption_batch_size,
        args.device,
        args.max_caption_images_per_post,
        args.force,
    )
    print("image2text done")
    image2vec(
        args.dataset_path,
        image_root,
        image_zip,
        args.image_batch_size,
        args.device,
        args.max_embedding_images_per_post,
        args.force,
    )
    print("image2vec done")
    merged_text_create_and_to_vec(args.dataset_path, args.text_batch_size, args.device, args.force)
    print("merged_text_create_and_to_vec done")
    merged_text2nouns2verb2adj(args.dataset_path, args.spacy_batch_size, args.force)
    print("merged_text2nouns2verb2adj done")
    print(f"Runtime: {(time.time() - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
