from pathlib import Path
from src.tissue_image import (
    TissueImage,
    crop_pair_to_true_correspondence,
    image_to_patch_grid,
)
from src.patchify import has_cluster
from saver import save_patch_pairs_for_octdiff
"""
To do:


- make sure patchify returns the relevant coordinates of the ST image. pair this wil HE image.
- treat each pair of patches as a single data point for training.
- apply the SR Model from OCTDiff onto the data (split into train, val, test)
- if num of patches is too small, data augmentation or smaller patch size.




/Dataset_Root
├── /Train 160/200
│ ├── /LowRes -> HE patches: []
│ └── /HiRes -> ST patches: [] << indexs of HE patches correspond to ST patches indexs
├── /Val 20/200
│ ├── /LowRes
│ └── /HiRes
└── /Test 20/200
│ ├── /LowRes
│ └── /HiRes




"""

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data/HE_ST_Dataset"
PATCH_SIZE = 8


def load_all_pngs(folder: Path):
    return sorted(folder.glob("*.png"))


def load_tissue_images(folder: Path, min_area: int = 600):
    if not folder.exists():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")

    tissue_images = []
    for path in load_all_pngs(folder):
        tissue_img = TissueImage(path)
        tissue_img.segment_tissue(min_area=min_area)
        tissue_images.append(tissue_img)

    return tissue_images


def split_he_st_images(images):
    he_objs = []
    st_objs = []

    for img in images:
        if img.name.endswith("HE"):
            he_objs.append(img)
        else:
            st_objs.append(img)

    return he_objs, st_objs


def build_aligned_pairs(he_objs, st_objs):
    aligned_pairs = []

    for he_img_obj, st_img_obj in zip(he_objs, st_objs):
        if len(he_img_obj.segments) != len(st_img_obj.segments):
            print(
                f"Warning: {he_img_obj.name} has {len(he_img_obj.segments)} segments "
                f"but {st_img_obj.name} has {len(st_img_obj.segments)} segments"
            )

        num_pairs = min(len(he_img_obj.segments), len(st_img_obj.segments))

        for segment_idx in range(num_pairs):
            he_seg = he_img_obj.segments[segment_idx]
            st_seg = st_img_obj.segments[segment_idx]

            try:
                he_crop, st_crop, common_bbox = crop_pair_to_true_correspondence(
                    he_img_obj, st_img_obj, he_seg, st_seg
                )

                aligned_pairs.append(
                    {
                        "he_name": he_img_obj.name,
                        "st_name": st_img_obj.name,
                        "segment_idx": segment_idx,
                        "he_crop": he_crop,
                        "st_crop": st_crop,
                        "common_bbox": common_bbox,
                    }
                )

            except ValueError as e:
                print(f"Skipping {st_img_obj.name} segment {segment_idx}: {e}")

    return aligned_pairs


def extract_patch_pairs_from_aligned_pair(pair, patch_size: int = 8, save_outputs: bool = True):
    he_crop = pair["he_crop"]
    st_crop = pair["st_crop"]

    try:
        he_grid = image_to_patch_grid(he_crop, patch_size=patch_size)
        st_grid = image_to_patch_grid(st_crop, patch_size=patch_size)
    except ValueError as e:
        print(
            f"Skipping {pair['st_name']} segment {pair['segment_idx']} because crop is too small: {e}"
        )
        return []

    cluster_result = has_cluster(
        st_crop,
        patch_size=patch_size,
        file_stem=f"{pair['st_name']}_segment_{pair['segment_idx']}",
        save_outputs=save_outputs,
    )

    pixel_match = cluster_result["pixel_match"]
    kept_patch_pairs = []

    rows = len(st_grid)
    cols = len(st_grid[0])

    for row in range(rows):
        for col in range(cols):
            y0 = row * patch_size
            y1 = y0 + patch_size
            x0 = col * patch_size
            x1 = x0 + patch_size

            if pixel_match[y0:y1, x0:x1].any():
                kept_patch_pairs.append(
                    {
                        "he_name": pair["he_name"],
                        "st_name": pair["st_name"],
                        "segment_idx": pair["segment_idx"],
                        "patch_row": row,
                        "patch_col": col,
                        "common_bbox": pair["common_bbox"],
                        "he_patch": he_grid[row][col],
                        "st_patch": st_grid[row][col],
                    }
                )

    return kept_patch_pairs


def build_patch_pairs(aligned_pairs, patch_size: int = 8, save_outputs: bool = True):
    all_patch_pairs = []

    for pair in aligned_pairs:
        patch_pairs = extract_patch_pairs_from_aligned_pair(
            pair,
            patch_size=patch_size,
            save_outputs=save_outputs,
        )
        all_patch_pairs.extend(patch_pairs)

    return all_patch_pairs


def main():
    images = load_tissue_images(DATA_DIR)
    he_objs, st_objs = split_he_st_images(images)
    aligned_pairs = build_aligned_pairs(he_objs, st_objs)
    patch_pairs = build_patch_pairs(
        aligned_pairs,
        patch_size=PATCH_SIZE,
        save_outputs=True,
    )

    print(f"Aligned segment pairs: {len(aligned_pairs)}")
    print(f"Kept patch pairs: {len(patch_pairs)}")

    dataset_root = BASE_DIR / "Dataset_Root"
    save_patch_pairs_for_octdiff(patch_pairs, dataset_root=dataset_root, seed=42)

    return patch_pairs


if __name__ == "__main__":
    patch_pairs = main()