import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
import torch
from angle_emb import AnglE
from PIL import Image
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor, ViTImageProcessor, ViTModel


DEFAULT_DATASET_PATH = Path("datasets/ICIP/dataset.pkl")
DEFAULT_IMAGE_ROOT = Path("datasets/raw_dataset/ICIP/pic")
LEGACY_IMAGE_ROOT = Path("datasets/origin_dataset/ICIP/pic")


def _batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield start, items[start:start + batch_size]


def _batch_count(items, batch_size):
    return (len(items) + batch_size - 1) // batch_size


def _column_ready(dataframe, columns):
    return all(column in dataframe.columns and len(dataframe[column]) == len(dataframe) for column in columns)


def _find_image_root(image_root=None):
    if image_root:
        path = Path(image_root)
        if not path.exists():
            raise FileNotFoundError(f"Image root not found: {path}")
        return path

    for path in [DEFAULT_IMAGE_ROOT, LEGACY_IMAGE_ROOT]:
        if path.exists():
            return path
    raise FileNotFoundError(
        "ICIP image directory was not found. Expected datasets/raw_dataset/ICIP/pic "
        "or pass --image_root."
    )


def _image_path(image_root, image_id):
    image_id = str(image_id)
    candidates = [
        Path(image_root) / f"{image_id}.jpg",
        Path(image_root) / image_id,
    ]
    if not image_id.lower().endswith(".jpg"):
        candidates.append(Path(image_root) / f"{image_id}.jpeg")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _open_image(image_root, image_id):
    path = _image_path(image_root, image_id)
    if path is None:
        return None
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except Exception:
        return None


def _write_missing_images(dataset_path, stage, missing_ids):
    if not missing_ids:
        return
    output_path = Path(dataset_path).with_name(f"missing_images_{stage}.txt")
    with open(output_path, "w", encoding="utf-8") as handle:
        for image_id in missing_ids:
            handle.write(f"{image_id}\n")
    print(f"Wrote missing image ids to {output_path}")


def _zero_embedding():
    return [[0.0] * 768]


def image2text(meta_file_path, image_root=None, batch_size=16, device=None, force=False):
    meta_data = pd.read_pickle(meta_file_path)
    if not force and _column_ready(meta_data, ["image_to_text"]):
        print("image_to_text already exists; skipping image captioning.")
        return []

    image_root = _find_image_root(image_root)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", use_fast=True)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)
    model.eval()

    image_ids = meta_data["image_id"].astype(str).tolist()
    image_to_text_list = []
    missing_ids = []

    for _, batch_ids in tqdm(_batched(image_ids, batch_size), total=_batch_count(image_ids, batch_size)):
        captions = ["0"] * len(batch_ids)
        images = []
        image_positions = []

        for batch_pos, image_id in enumerate(batch_ids):
            image = _open_image(image_root, image_id)
            if image is None:
                missing_ids.append(image_id)
                continue
            images.append(image)
            image_positions.append(batch_pos)

        if images:
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            with torch.inference_mode():
                generated_ids = model.generate(**inputs)
            decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
            for batch_pos, text in zip(image_positions, decoded):
                captions[batch_pos] = text

        image_to_text_list.extend(captions)

    meta_data["image_to_text"] = image_to_text_list
    meta_data.to_pickle(meta_file_path)
    _write_missing_images(meta_file_path, "caption", missing_ids)
    print(f"image2text done; missing or unreadable images: {len(missing_ids)}")
    return missing_ids


def image2vec(meta_file_path, image_root=None, batch_size=64, device=None, force=False):
    meta_data = pd.read_pickle(meta_file_path)
    if not force and _column_ready(meta_data, ["mean_pooling_vec", "cls_vec"]):
        print("Image embeddings already exist; skipping image2vec.")
        return []

    image_root = _find_image_root(image_root)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
    model.eval()

    image_ids = meta_data["image_id"].astype(str).tolist()
    mean_pooling_vec_list = []
    cls_vec_list = []
    missing_ids = []

    for _, batch_ids in tqdm(_batched(image_ids, batch_size), total=_batch_count(image_ids, batch_size)):
        mean_batch = [_zero_embedding() for _ in batch_ids]
        cls_batch = [_zero_embedding() for _ in batch_ids]
        images = []
        image_positions = []

        for batch_pos, image_id in enumerate(batch_ids):
            image = _open_image(image_root, image_id)
            if image is None:
                missing_ids.append(image_id)
                continue
            images.append(image)
            image_positions.append(batch_pos)

        if images:
            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.inference_mode():
                outputs = model(**inputs)
            cls_output = outputs.last_hidden_state[:, 0, :].detach().cpu()
            mean_output = outputs.last_hidden_state.mean(dim=1).detach().cpu()

            for encoded_pos, batch_pos in enumerate(image_positions):
                mean_batch[batch_pos] = mean_output[encoded_pos:encoded_pos + 1].tolist()
                cls_batch[batch_pos] = cls_output[encoded_pos:encoded_pos + 1].tolist()

        mean_pooling_vec_list.extend(mean_batch)
        cls_vec_list.extend(cls_batch)

    meta_data["mean_pooling_vec"] = mean_pooling_vec_list
    meta_data["cls_vec"] = cls_vec_list
    meta_data.to_pickle(meta_file_path)
    _write_missing_images(meta_file_path, "embedding", missing_ids)
    print(f"image2vec done; missing or unreadable images: {len(missing_ids)}")
    return missing_ids


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


def validate_image_features(meta_file_path):
    meta_data = pd.read_pickle(meta_file_path)
    required_columns = ["image_to_text", "mean_pooling_vec", "cls_vec"]
    missing = [column for column in required_columns if column not in meta_data.columns]
    if missing:
        raise KeyError(f"ICIP preprocessing did not create required columns: {missing}")

    captions = meta_data["image_to_text"].fillna("").astype(str).str.strip()
    zero_caption_count = int(captions.isin(["", "0"]).sum())
    mean_vec = np.asarray(meta_data["mean_pooling_vec"].tolist(), dtype=np.float32)
    cls_vec = np.asarray(meta_data["cls_vec"].tolist(), dtype=np.float32)

    if zero_caption_count == len(meta_data) and np.allclose(mean_vec, 0.0) and np.allclose(cls_vec, 0.0):
        raise RuntimeError(
            "All ICIP image captions and image embeddings are empty. Check --image_root "
            "or rerun this script with --force after fixing the image path."
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Extract ICIP image, text, and POS features.")
    parser.add_argument("--dataset_path", default=str(DEFAULT_DATASET_PATH), help="ICIP dataset pickle path")
    parser.add_argument("--image_root", default=None, help="Directory containing ICIP images")
    parser.add_argument("--device", default=None, help="Torch device, for example cuda:0 or cpu")
    parser.add_argument("--caption_batch_size", default=16, type=int, help="BLIP captioning batch size")
    parser.add_argument("--image_batch_size", default=64, type=int, help="ViT image embedding batch size")
    parser.add_argument("--text_batch_size", default=256, type=int, help="AnglE text embedding batch size")
    parser.add_argument("--spacy_batch_size", default=256, type=int, help="spaCy pipe batch size")
    parser.add_argument("--force", action="store_true", help="Recompute features even if output columns exist")
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    image2text(args.dataset_path, args.image_root, args.caption_batch_size, args.device, args.force)
    image2vec(args.dataset_path, args.image_root, args.image_batch_size, args.device, args.force)
    merged_text_create_and_to_vec(args.dataset_path, args.text_batch_size, args.device, args.force)
    print("merged_text_create_and_to_vec done")
    merged_text2nouns2verb2adj(args.dataset_path, args.spacy_batch_size, args.force)
    print("merged_text2nouns2verb2adj done")
    validate_image_features(args.dataset_path)
    print(f"Runtime: {(time.time() - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
