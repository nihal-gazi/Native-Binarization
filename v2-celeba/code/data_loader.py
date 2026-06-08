import os
import zipfile
import random
import glob
import csv
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision.utils import save_image
from .config import DATA_DIR, SAMPLE_DIR, IMAGE_SIZE
from .logger import logger

class CelebAZipDataset(Dataset):
    """Custom dataset that indexes aligned, filtered grayscale CelebA images."""
    def __init__(self, image_paths, landmarks_dict, img_size=IMAGE_SIZE):
        self.image_paths = image_paths
        self.landmarks_dict = landmarks_dict
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image_id = os.path.basename(path)
        
        # Load image with OpenCV (directly as grayscale)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback in case of read error
            img = np.zeros((178, 178), dtype=np.uint8) + 255
            
        # Align image using landmarks similarity transform
        if image_id in self.landmarks_dict:
            lx, ly, rx, ry = self.landmarks_dict[image_id]
            p_in = np.array([[lx, ly], [rx, ry]], dtype=np.float32)
            # Target eye positions in square crop
            p_out = np.array([
                [self.img_size * 0.35, self.img_size * 0.40],
                [self.img_size * 0.65, self.img_size * 0.40]
            ], dtype=np.float32)
            
            M, _ = cv2.estimateAffinePartial2D(p_in, p_out)
            # Warp to self.img_size x self.img_size with cubic interpolation and white background border (255)
            aligned = cv2.warpAffine(img, M, (self.img_size, self.img_size),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=255)
        else:
            # Fallback to strict center crop and resize
            h, w = img.shape
            crop_size = min(h, w)
            start_y = (h - crop_size) // 2
            start_x = (w - crop_size) // 2
            cropped = img[start_y:start_y+crop_size, start_x:start_x+crop_size]
            aligned = cv2.resize(cropped, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
            
        # Convert to tensor and normalize to [-1, 1]
        tensor = torch.from_numpy(aligned).float() / 255.0
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.unsqueeze(0)  # Shape (1, H, W)
        
        return tensor, 0

def check_and_extract_filtered_celeba(target_count=10000):
    """Parses attributes/landmarks, extracts and filters CelebA images directly from zip."""
    raw_dir = os.path.join(DATA_DIR, "raw")
    zip_path = os.path.join(raw_dir, "img_align_celeba.zip")
    extract_target = os.path.join(raw_dir, "img_align_celeba")
    os.makedirs(extract_target, exist_ok=True)
    
    # 1. Error if raw zip is missing
    if not os.path.exists(zip_path):
        err_msg = (
            f"\n\n[ERROR] img_align_celeba.zip not found at: {zip_path}\n"
            "Please manually place 'img_align_celeba.zip' inside the v2-celeba/data/raw/ directory.\n"
        )
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)
        
    # 2. Extract list files if not present
    attr_path = os.path.join(raw_dir, "list_attr_celeba.csv")
    landmarks_path = os.path.join(raw_dir, "list_landmarks_align_celeba.csv")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        if not os.path.exists(attr_path):
            logger.info("Extracting list_attr_celeba.csv...")
            zip_ref.extract("list_attr_celeba.csv", raw_dir)
        if not os.path.exists(landmarks_path):
            logger.info("Extracting list_landmarks_align_celeba.csv...")
            zip_ref.extract("list_landmarks_align_celeba.csv", raw_dir)
            
    # 3. Read and filter by attributes
    logger.info("Parsing attribute and landmark lists...")
    with open(attr_path, "r", encoding="utf-8") as f:
        attr_reader = list(csv.DictReader(f))
        
    with open(landmarks_path, "r", encoding="utf-8") as f:
        landmarks_reader = list(csv.DictReader(f))
        
    landmarks_dict = {}
    for row in landmarks_reader:
        landmarks_dict[row["image_id"]] = [
            float(row["lefteye_x"]), float(row["lefteye_y"]),
            float(row["righteye_x"]), float(row["righteye_y"])
        ]
        
    # Keep only No_Beard == 1 and Straight_Hair == 1 for structural homogeneity
    attr_filtered = [
        row["image_id"] for row in attr_reader
        if int(row.get("No_Beard", 0)) == 1 and int(row.get("Straight_Hair", 0)) == 1
    ]
    
    logger.info(f"Attribute filter passed: {len(attr_filtered)} candidate images.")
    
    # 4. Check target directory for already extracted and verified images
    existing_paths = glob.glob(os.path.join(extract_target, "*.jpg"))
    existing_ids = {os.path.basename(p) for p in existing_paths}
    
    kept_paths = []
    # Build list of already matching images
    for img_id in attr_filtered:
        img_out_path = os.path.join(extract_target, img_id)
        if img_id in existing_ids:
            kept_paths.append(img_out_path)
            
    # If we already have enough kept images, return them immediately
    if len(kept_paths) >= target_count:
        logger.info(f"Found {len(kept_paths)} pre-filtered images cached on disk. Reusing.")
        return kept_paths[:target_count], landmarks_dict
        
    # Otherwise, extract more from zip and verify background corners
    logger.info(f"Cached verified images ({len(kept_paths)}) below target ({target_count}). Scanning and extracting more...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for img_id in attr_filtered:
            if len(kept_paths) >= target_count:
                break
                
            img_out_path = os.path.join(extract_target, img_id)
            if img_id in existing_ids:
                continue
                
            try:
                # Extract image file
                zip_ref.extract("img_align_celeba/img_align_celeba/" + img_id, raw_dir)
                
                # Move to flat target structure
                extracted_src = os.path.join(raw_dir, "img_align_celeba", "img_align_celeba", img_id)
                if os.path.exists(extracted_src):
                    # Check background corners
                    img = cv2.imread(extracted_src, cv2.IMREAD_GRAYSCALE)
                    corners = [img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]]
                    
                    if np.mean(corners) > 150:  # Light/white background corner threshold
                        os.rename(extracted_src, img_out_path)
                        kept_paths.append(img_out_path)
                    else:
                        os.remove(extracted_src)  # Discard dark background image
            except Exception as e:
                logger.warning(f"Error processing {img_id}: {e}")
                
    # Clean up empty zip folder structures
    nested_dir = os.path.join(raw_dir, "img_align_celeba", "img_align_celeba")
    if os.path.exists(nested_dir):
        try:
            os.rmdir(nested_dir)
        except:
            pass
            
    logger.info(f"Verified dataset prepared. Total aligned and filtered images on disk: {len(kept_paths)}")
    return kept_paths[:target_count], landmarks_dict

def get_celeba_dataset(truncate_size=None):
    """Initializes the double-filtered aligned dataset."""
    target_count = truncate_size if truncate_size is not None else 10000
    image_paths, landmarks_dict = check_and_extract_filtered_celeba(target_count=target_count)
    
    dataset = CelebAZipDataset(image_paths, landmarks_dict)
    
    if truncate_size is not None and len(dataset) > truncate_size:
        random.seed(42)
        indices = random.sample(range(len(dataset)), truncate_size)
        dataset = Subset(dataset, indices)
        
    return dataset

def save_sample_grid():
    """Generates and saves the visual verification grid."""
    dataset = get_celeba_dataset(truncate_size=32)
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch, _ = next(iter(loader))
    
    # Scale from [-1, 1] back to [0, 1]
    grid_img = (batch + 1.0) / 2.0
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    out_path = os.path.join(SAMPLE_DIR, "preprocessed_samples.png")
    save_image(grid_img, out_path, nrow=8)
    logger.info(f"Phase 1: Pre-processed face-aligned grayscale samples saved directly to {out_path}")

if __name__ == "__main__":
    save_sample_grid()
