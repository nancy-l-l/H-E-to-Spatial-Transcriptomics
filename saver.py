from pathlib import Path
import random
import cv2

def save_patch_pairs_for_octdiff(
    patch_pairs,
    dataset_root: Path,
    seed: int = 42,
):
    """
    Save patch pairs into:
        Dataset_Root/
            Train/LowRes
            Train/HiRes
            Val/LowRes
            Val/HiRes
            Test/LowRes
            Test/HiRes

    Each HE/ST patch pair is saved with the same filename stem so the model
    can learn corresponding pairs.

    Expected patch_pairs format:
        [
            {
                "he_name": str,
                "st_name": str,
                "segment_idx": int,
                "patch_row": int,
                "patch_col": int,
                "he_patch": np.ndarray,   # BGR image
                "st_patch": np.ndarray,   # BGR image
                ...
            },
            ...
        ]
    """
    dataset_root = Path(dataset_root)

    split_dirs = {
        "Train": {
            "LowRes": dataset_root / "Train" / "LowRes",
            "HiRes": dataset_root / "Train" / "HiRes",
        },
        "Val": {
            "LowRes": dataset_root / "Val" / "LowRes",
            "HiRes": dataset_root / "Val" / "HiRes",
        },
        "Test": {
            "LowRes": dataset_root / "Test" / "LowRes",
            "HiRes": dataset_root / "Test" / "HiRes",
        },
    }

    for split in split_dirs.values():
        split["LowRes"].mkdir(parents=True, exist_ok=True)
        split["HiRes"].mkdir(parents=True, exist_ok=True)

    shuffled = patch_pairs.copy()
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * 0.8)
    n_test = int(n * 0.1)
    n_val = n - n_train - n_test

    split_data = {
        "Train": shuffled[:n_train],
        "Test": shuffled[n_train:n_train + n_test],
        "Val": shuffled[n_train + n_test:],
    }

    print(f"Total patch pairs: {n}")
    print(f"Train: {len(split_data['Train'])}")
    print(f"Test: {len(split_data['Test'])}")
    print(f"Val: {len(split_data['Val'])}")

    for split_name, items in split_data.items():
        lowres_dir = split_dirs[split_name]["LowRes"]
        hires_dir = split_dirs[split_name]["HiRes"]

        for idx, pair in enumerate(items):
            sample_id = (
                f"{pair['st_name']}_seg{pair['segment_idx']:03d}"
                f"_r{pair['patch_row']:03d}_c{pair['patch_col']:03d}"
            )

            he_out = lowres_dir / f"{sample_id}.png"
            st_out = hires_dir / f"{sample_id}.png"

            he_patch = pair["he_patch"]
            st_patch = pair["st_patch"]

            ok1 = cv2.imwrite(str(he_out), he_patch)
            ok2 = cv2.imwrite(str(st_out), st_patch)

            if not ok1 or not ok2:
                raise IOError(f"Failed to save patch pair: {sample_id}")

    print(f"Saved OCTDiff-formatted dataset to: {dataset_root}")