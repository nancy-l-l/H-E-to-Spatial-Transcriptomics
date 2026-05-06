'''
from pathlib import Path
import csv
import os

def create_patch_weights_csv(dataset_root: str, output_csv: str, split: str = "train", hires_dir_name: str = "HiRes"):
    """
    Build a weights CSV for OCTDiff from filenames in Dataset_Root/<split>/<hires_dir_name>.

    The current OCTDiff loader uses the FIRST 7 CHARACTERS of each HiRes filename
    as the lookup key, so this script writes:
        <filename[:7]>,1.0

    Example:
        abc1234_seg001_r000_c000.png  ->  abc1234,1.0
    """
    dataset_root = Path(dataset_root)
    hires_dir = dataset_root / split / hires_dir_name

    if not hires_dir.exists():
        raise FileNotFoundError(f"HiRes directory not found: {hires_dir}")

    names = sorted(
        p.stem for p in hires_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )

    prefixes = sorted({name[:7] for name in names if len(name) >= 7})

    if not prefixes:
        raise ValueError(f"No image filenames found in {hires_dir}")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        for prefix in prefixes:
            writer.writerow([prefix, 1.0])

    print(f"Wrote {len(prefixes)} rows to {output_csv}")

if __name__ == "__main__":
    # Update these if needed
    create_patch_weights_csv(
        dataset_root="Dataset_Root",
        output_csv="data/patch_weights.csv",
        split="train",
        hires_dir_name="HiRes",
    )
'''
import csv
from pathlib import Path

def create_patch_weights_csv(dataset_root, output_csv, split="train", hires_dir_name="B"):
    hires_dir = Path(dataset_root) / split / hires_dir_name

    names = sorted(
        p.stem for p in hires_dir.glob("*.png")
    )

    if not names:
        raise ValueError(f"No images found in {hires_dir}")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        for name in names:
            writer.writerow([name, 1.0])

    print(f"Wrote {len(names)} rows to {output_csv}")

if __name__ == "__main__":
    # Update these if needed
    create_patch_weights_csv(
        dataset_root="Dataset_Root",
        output_csv="data/patch_weights.csv",
        split="train",
        hires_dir_name="HiRes",
    )