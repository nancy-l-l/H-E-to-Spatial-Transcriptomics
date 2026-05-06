import os
import shutil
import random

def create_directories(output_dir):
    """Create train, val, and test directories for A and B."""
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'A'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'B'), exist_ok=True)

def get_image_pairs(input_dir_A, input_dir_B):
    """Get sorted lists of image pairs from two directories."""
    images_A = sorted([f for f in os.listdir(input_dir_A) if f.endswith('.bmp')])
    images_B = sorted([f for f in os.listdir(input_dir_B) if f.endswith('.bmp')])

    assert len(images_A) == len(images_B), "Mismatch in the number of images between A and B directories."

    return list(zip(images_A, images_B))

def split_dataset(paired_images, train_split=0.8, val_split=0.1):
    """Split the dataset into train, val, and test sets."""
    random.shuffle(paired_images)

    train_size = int(train_split * len(paired_images))
    val_size = int(val_split * len(paired_images))

    train_pairs = paired_images[:train_size]
    val_pairs = paired_images[train_size:train_size + val_size]
    test_pairs = paired_images[train_size + val_size:]

    return train_pairs, val_pairs, test_pairs

def copy_pairs(pairs, input_dir_A, input_dir_B, folder_A, folder_B):
    """Copy image pairs to the specified directories."""
    for file_A, file_B in pairs:
        shutil.copy(os.path.join(input_dir_A, file_A), os.path.join(folder_A, file_A))
        shutil.copy(os.path.join(input_dir_B, file_B), os.path.join(folder_B, file_B))

def main():
    input_dir_A = '/data/After Register Low'  #CHANGE TO YOUR PATH
    input_dir_B = '/data/After Register High' #CHANGE TO YOUR PATH
    output_dir = '/data/Registered pOCT Dataset' #CHANGE TO YOUR PATH
 
    os.makedirs(output_dir, exist_ok=True)
    create_directories(output_dir)

    paired_images = get_image_pairs(input_dir_A, input_dir_B)

    train_pairs, val_pairs, test_pairs = split_dataset(paired_images)    

    copy_pairs(train_pairs, input_dir_A, input_dir_B, os.path.join(output_dir, 'train', 'A'), os.path.join(output_dir, 'train', 'B'))
    print(f"Copied {len(train_pairs)} training pairs.")
    copy_pairs(val_pairs, input_dir_A, input_dir_B, os.path.join(output_dir, 'val', 'A'), os.path.join(output_dir, 'val', 'B'))
    print(f"Copied {len(val_pairs)} validation pairs.")
    copy_pairs(test_pairs, input_dir_A, input_dir_B, os.path.join(output_dir, 'test', 'A'), os.path.join(output_dir, 'test', 'B'))
    print(f"Copied {len(test_pairs)} test pairs.")

if __name__ == "__main__":
    main()