from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path
from typing import Union, Optional
import cv2


cluster_colors = np.array([
    (44, 19, 57),
    (109, 61, 143),
    (191, 247, 93),
    (57, 47, 122),
    (208, 237, 90),
    (68, 74, 173),
    (223, 222, 89),
    (77, 99, 212),
    (234, 205, 88),
    (83, 124, 236),
    (138, 128, 173),
    (241, 185, 83),
    (88, 148, 247),
    (241, 162, 73),
    (90, 173, 239),
    (235, 134, 61),
    (94, 195, 221),
    (226, 108, 49),
    (102, 215, 198),
    (213, 85, 40),
    (110, 229, 172),
    (196, 65, 36),
    (106, 238, 145),
    (173, 47, 36),
    (95, 243, 118),
    (148, 31, 37),
    (77, 244, 94),
    (120, 18, 35),
], dtype=np.int16)


def _load_input_image(st_img: Union[str, Path, np.ndarray]):
    if isinstance(st_img, (str, Path)):
        img_path = Path(st_img)
        pil_img = Image.open(img_path).convert("RGB")
        arr_rgb = np.array(pil_img, dtype=np.int16)
        base_name = img_path.stem
        return pil_img, arr_rgb, base_name

    if isinstance(st_img, np.ndarray):
        if st_img.ndim != 3 or st_img.shape[2] != 3:
            raise ValueError(f"Expected image array of shape (H, W, 3), got {st_img.shape}")

        # cv2 images are BGR, convert to RGB for PIL and color matching
        arr_uint8 = st_img.astype(np.uint8)
        arr_rgb_uint8 = cv2.cvtColor(arr_uint8, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(arr_rgb_uint8)
        arr_rgb = arr_rgb_uint8.astype(np.int16)
        base_name = "st_crop"
        return pil_img, arr_rgb, base_name

    raise TypeError(
        "has_cluster expected a file path or numpy image array, "
        f"but got {type(st_img)}"
    )


def has_cluster(
    st_img: Union[str, Path, np.ndarray],
    patch_size: int = 8,
    tolerance: int = 35,
    min_channel_range: int = 40,
    save_outputs: bool = True,
    out_dir: Union[str, Path, None] = None,
    file_stem: Optional[str] = None,
):
    pil_img, target, detected_base_name = _load_input_image(st_img)

    base_name = file_stem if file_stem is not None else detected_base_name

    BASE_DIR = Path(__file__).resolve().parent.parent
    OUT_DIR = Path(out_dir) if out_dir is not None else BASE_DIR / "output" / "2B_deleted"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    diff = np.abs(target[:, :, None, :] - cluster_colors[None, None, :, :])
    close_to_cluster = diff.max(axis=3).min(axis=2) <= tolerance

    channel_range = target.max(axis=2) - target.min(axis=2)
    is_colorful = channel_range >= min_channel_range

    pixel_match = close_to_cluster & is_colorful

    mask_path = None
    out_path = None

    if save_outputs:
        mask_path = OUT_DIR / f"{base_name}_matched_pixels.png"
        mask_img = (pixel_match.astype(np.uint8) * 255)
        Image.fromarray(mask_img).save(mask_path)

    result = pil_img.copy()
    draw = ImageDraw.Draw(result)

    height, width = pixel_match.shape
    num_boxes = 0

    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch = pixel_match[y:min(y + patch_size, height), x:min(x + patch_size, width)]
            if patch.any():
                x1 = min(x + patch_size - 1, width - 1)
                y1 = min(y + patch_size - 1, height - 1)
                draw.rectangle([x, y, x1, y1], outline=(255, 0, 0), width=1)
                num_boxes += 1

    if save_outputs:
        out_path = OUT_DIR / f"{base_name}_patch_highlighted_{patch_size}x{patch_size}.png"
        result.save(out_path)

    print(f"Processed: {base_name}")
    if out_path is not None:
        print(f"Saved highlighted image to: {out_path}")
    if mask_path is not None:
        print(f"Saved matched-pixel mask to: {mask_path}")
    print(f"Matched pixels: {int(pixel_match.sum())}")
    print(f"Boxes drawn: {num_boxes}")
    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"Tolerance used: {tolerance}")
    print(f"Minimum channel range: {min_channel_range}")
    print("-" * 50)

    return {
        "pixel_match": pixel_match,
        "matched_pixels": int(pixel_match.sum()),
        "num_boxes": num_boxes,
        "mask_path": mask_path,
        "out_path": out_path,
        "patch_size": patch_size,
        "tolerance": tolerance,
        "min_channel_range": min_channel_range,
    }
