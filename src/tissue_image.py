from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple, List

class TissueSegment:
   def __init__(self, segment_id: int, contour: np.ndarray, bbox: Tuple[int, int, int, int]):
       self.segment_id = segment_id
       self.contour = contour
       self.bbox = bbox


   @property
   def coordinates(self):
       return self.contour.tolist()


   def __repr__(self):
       x, y, w, h = self.bbox
       return f"TissueSegment(id={self.segment_id}, bbox=({x}, {y}, {w}, {h}), points={len(self.contour)})"

class TissueImage:
   def __init__(self, image_path: Union[str, Path]):
       self.image_path = Path(image_path)
       self.name = self.image_path.stem


       self.image_bgr = cv2.imread(str(self.image_path))
       if self.image_bgr is None:
           raise ValueError(f"Could not read image: {self.image_path}")


       self.mask = None
       self.segments: List["TissueSegment"] = []


   def segment_tissue(self, min_area: int = 600):
       image = self.image_bgr


       hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
       gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


       h = hsv[:, :, 0]
       s = hsv[:, :, 1]
       v = hsv[:, :, 2]


       candidate_mask = (
           ((gray < 242) & (s > 8)) |
           ((gray < 230) & (s > 4)) |
           ((v < 245) & (s > 6))
       ).astype(np.uint8)


       kernel_open = np.ones((3, 3), np.uint8)
       kernel_close = np.ones((9, 9), np.uint8)


       cleaned = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel_open)
       cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)


       contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


       final_mask = np.zeros_like(cleaned)
       segments = []
       seg_id = 0


       for contour in contours:
           area = cv2.contourArea(contour)
           if area < min_area:
               print('skipped')
               continue


           x, y, w, h = cv2.boundingRect(contour)


           if w < 30 or h < 8:
               continue


           cv2.drawContours(final_mask, [contour], -1, 1, thickness=cv2.FILLED)


           contour_points = contour.squeeze(axis=1)
           segments.append(TissueSegment(seg_id, contour_points, (x, y, w, h)))
           seg_id += 1


       self.mask = final_mask
       self.segments = sorted(segments, key=lambda seg: (seg.bbox[1], seg.bbox[0]))


   def get_segment_coordinates(self):
       return [segment.coordinates for segment in self.segments]


   def get_patch_images(self, pad: int = 10):
       patches = []
       H, W = self.image_bgr.shape[:2]


       for segment in self.segments:
           x, y, w, h = segment.bbox
           x0 = max(0, x - pad)
           y0 = max(0, y - pad)
           x1 = min(W, x + w + pad)
           y1 = min(H, y + h + pad)


           patch = self.image_bgr[y0:y1, x0:x1].copy()
           patches.append((segment.segment_id, patch))


       return patches


   def crop_from_bbox(self, bbox):
       """
       Crop from the original image using bbox = (x, y, w, h).
       """
       x, y, w, h = bbox
       return self.image_bgr[y:y+h, x:x+w].copy()


   def show_patches(self):
       patches = self.get_patch_images()


       if not patches:
           print(f"{self.name}: no tissue patches found")
           return


       cols = min(3, len(patches))
       rows = (len(patches) + cols - 1) // cols


       plt.figure(figsize=(5 * cols, 4 * rows))
       for i, (segment_id, patch) in enumerate(patches):
           plt.subplot(rows, cols, i + 1)
           plt.imshow(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
           plt.title(f"{self.name} - patch {segment_id}")
           plt.axis("off")
       plt.tight_layout()
       plt.show()


   def print_segment_info(self):
       print(f"\nImage: {self.name}")
       print(f"Detected {len(self.segments)} tissue segment(s)")
       for segment in self.segments:
           print(segment)
           print("First 10 contour coordinates:", segment.coordinates[:10])




def intersection_bbox(bbox1, bbox2):
   """
   Return the overlapping rectangle between two bboxes in the same
   full-image coordinate system.


   Input bboxes are (x, y, w, h).
   Output is also (x, y, w, h).
   """
   x1, y1, w1, h1 = bbox1
   x2, y2, w2, h2 = bbox2


   left = max(x1, x2)
   top = max(y1, y2)
   right = min(x1 + w1, x2 + w2)
   bottom = min(y1 + h1, y2 + h2)


   if right <= left or bottom <= top:
       raise ValueError(f"No overlap between bboxes: {bbox1} and {bbox2}")


   return (left, top, right - left, bottom - top)




def crop_pair_to_true_correspondence(he_tissue_img, st_tissue_img, he_segment, st_segment):
   """
   Create truly corresponding HE/ST crops by using the intersection
   of the paired segment bounding boxes.
   """
   common_bbox = intersection_bbox(he_segment.bbox, st_segment.bbox)


   he_crop = he_tissue_img.crop_from_bbox(common_bbox)
   st_crop = st_tissue_img.crop_from_bbox(common_bbox)


   return he_crop, st_crop, common_bbox

def image_to_patch_grid(image: np.ndarray, patch_size: int = 8):
    h, w = image.shape[:2]

    rows = h // patch_size
    cols = w // patch_size

    if rows == 0 or cols == 0:
        raise ValueError(
            f"Image is too small for {patch_size}x{patch_size} patches. "
            f"Image shape: {image.shape}"
        )

    usable_h = rows * patch_size
    usable_w = cols * patch_size
    cropped = image[:usable_h, :usable_w]

    patch_grid = []
    for row in range(rows):
        row_patches = []
        for col in range(cols):
            y0 = row * patch_size
            y1 = y0 + patch_size
            x0 = col * patch_size
            x1 = x0 + patch_size
            row_patches.append(cropped[y0:y1, x0:x1].copy())
        patch_grid.append(row_patches)

    return patch_grid

def show_pairs(pair):
   he_img, st_img = pair


   plt.figure(figsize=(10, 4))


   plt.subplot(1, 2, 1)
   plt.imshow(cv2.cvtColor(he_img, cv2.COLOR_BGR2RGB))
   plt.title(f"HE {he_img.shape[:2]}")
   plt.axis("off")


   plt.subplot(1, 2, 2)
   plt.imshow(cv2.cvtColor(st_img, cv2.COLOR_BGR2RGB))
   plt.title(f"ST {st_img.shape[:2]}")
   plt.axis("off")


   plt.tight_layout()
   plt.show()
