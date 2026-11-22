import shutil
import zipfile
from pathlib import Path

# ---------------- SETTINGS (edit these) ----------------
HUMAN_ZIP = "human.zip"        # zip whose class 0 = human
FIRE_ZIP = "fire.zip"          # zip whose class 0 = fire
OUTPUT_DIR = "merged_dataset"  # new folder that will be created

# old class id -> new class id, for each dataset
HUMAN_CLASS_MAP = {0: 0}       # human stays 0
FIRE_CLASS_MAP = {0: 1}        # fire 0 becomes 1

CLASS_NAMES = ["human", "fire"]
# -------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Different tools name the splits differently ("valid" vs "val")
SPLIT_NAMES = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}


def find_split(path):
    """Work out whether a file belongs to train, valid or test."""
    for part in path.parts:
        if part.lower() in SPLIT_NAMES:
            return SPLIT_NAMES[part.lower()]
    return "train"  # no split folder found -> put it in train


def find_label_file(image_path):
    """Find the .txt label for an image (the 'images' folder swapped for 'labels')."""
    parts = list(image_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def remap_label(label_path, class_map):
    """Read a label file and change the class ids. Returns new text + dropped count."""
    if not label_path.exists():
        return "", 0  # image with no objects (that's fine)

    new_lines, dropped = [], 0
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        old_id = int(float(parts[0]))
        if old_id not in class_map:
            dropped += 1
            continue
        parts[0] = str(class_map[old_id])
        new_lines.append(" ".join(parts))
    return "\n".join(new_lines) + ("\n" if new_lines else ""), dropped


def add_dataset(zip_name, prefix, class_map, out_dir, temp_dir):
    zip_path = Path(zip_name)
    if not zip_path.exists():
        raise SystemExit(f"ERROR: can't find '{zip_name}'. Is it in this folder?")

    extract_to = temp_dir / prefix
    print(f"Unzipping {zip_name} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_to)

    counts = {"train": 0, "valid": 0, "test": 0}
    total_dropped = 0

    for img in extract_to.rglob("*"):
        if img.suffix.lower() not in IMAGE_EXTS or "images" not in img.parts:
            continue

        split = find_split(img.relative_to(extract_to))
        new_name = f"{prefix}_{img.stem}"  # prefix avoids name clashes

        img_dest = out_dir / split / "images" / (new_name + img.suffix)
        lbl_dest = out_dir / split / "labels" / (new_name + ".txt")
        img_dest.parent.mkdir(parents=True, exist_ok=True)
        lbl_dest.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(img, img_dest)
        text, dropped = remap_label(find_label_file(img), class_map)
        lbl_dest.write_text(text)

        counts[split] += 1
        total_dropped += dropped

    print(f"  {prefix}: train={counts['train']}  valid={counts['valid']}  test={counts['test']}")
    if total_dropped:
        print(f"  WARNING: skipped {total_dropped} boxes with unexpected class ids")
    return counts


def write_data_yaml(out_dir, splits_used):
    lines = [
        f"path: {out_dir.resolve().as_posix()}",
        "train: train/images",
        "val: valid/images" if "valid" in splits_used else "val: train/images",
    ]
    if "test" in splits_used:
        lines.append("test: test/images")
    lines += ["", f"nc: {len(CLASS_NAMES)}", "names:"]
    lines += [f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)]
    (out_dir / "data.yaml").write_text("\n".join(lines) + "\n")


def main():
    out_dir = Path(OUTPUT_DIR)
    if out_dir.exists():
        raise SystemExit(f"ERROR: '{OUTPUT_DIR}' already exists. Delete or rename it first.")

    temp_dir = Path("_temp_unzip")
    shutil.rmtree(temp_dir, ignore_errors=True)

    try:
        a = add_dataset(HUMAN_ZIP, "human", HUMAN_CLASS_MAP, out_dir, temp_dir)
        b = add_dataset(FIRE_ZIP, "fire", FIRE_CLASS_MAP, out_dir, temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)  # clean up

    splits_used = {s for s in ("train", "valid", "test") if a[s] + b[s] > 0}
    if not splits_used:
        raise SystemExit("ERROR: no images found. Check your zip files have an 'images' folder.")
    if "valid" not in splits_used:
        print("NOTE: no validation images found, so 'val' points to the train images.")

    write_data_yaml(out_dir, splits_used)
    print(f"\nDone! Merged dataset is in: {out_dir.resolve()}")
    print(f"Your data.yaml: {(out_dir / 'data.yaml').resolve()}")


if __name__ == "__main__":
    main()