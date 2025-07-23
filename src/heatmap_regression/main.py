# Heatmap Regression Implementation for Human Pose Estimation on MS COCO
# This script provides a complete pipeline using the modern heatmap-based approach.
# 1. A model with a ResNet backbone and a deconvolutional head to predict heatmaps.
# 2. A custom PyTorch Dataset that generates ground-truth Gaussian heatmaps.
# 3. A training loop using MSE loss between predicted and ground-truth heatmaps.
# 4. An evaluation function that finds the coordinates from heatmaps for AP calculation.
# 5. A visualization function to draw the final pose.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms

import os
import json
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ..configuration.config import get_configuration

# ==============================================================================
# 1. MODEL DEFINITION (Heatmap Prediction)
# ==============================================================================
class HeatmapPoseModel(nn.Module):
    """
    Model that uses a ResNet backbone and a deconvolutional head to predict heatmaps.
    """
    def __init__(self, num_keypoints=17):
        super(HeatmapPoseModel, self).__init__()
        
        # Load a pre-trained ResNet-50 model
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Use the ResNet backbone up to the last convolutional block
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Define the deconvolutional head to upsample feature maps to heatmaps
        self.deconv_head = nn.Sequential(
            nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Final layer to produce the heatmaps
            nn.Conv2d(256, num_keypoints, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.deconv_head(x)
        return x

# ==============================================================================
# 2. COCO DATASET LOADER (Generates Heatmaps)
# ==============================================================================
class CocoHeatmapDataset(Dataset):
    """
    Custom PyTorch Dataset for MS COCO that generates ground-truth heatmaps.
    """
    def __init__(self, root_dir, annotation_file, transform=None, input_size=224, heatmap_size=56):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.transform = transform
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.stride = input_size / heatmap_size
        self.ids = self._get_image_ids()

    def _get_image_ids(self):
        ids = []
        cat_ids = self.coco.getCatIds(catNms=['person'])
        for img_id in self.coco.getImgIds():
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False)
            if ann_ids:
                anns = self.coco.loadAnns(ann_ids)
                if any(ann['num_keypoints'] > 0 for ann in anns):
                    ids.append(img_id)
        return ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.coco.getCatIds(catNms=['person']))
        anns = self.coco.loadAnns(ann_ids)
        
        largest_person_ann = None
        max_area = 0
        for ann in anns:
            if ann['num_keypoints'] > 0 and ann['area'] > max_area:
                max_area = ann['area']
                largest_person_ann = ann
        
        if largest_person_ann is None:
            return self.__getitem__((idx + 1) % len(self))
            
        bbox = largest_person_ann['bbox']
        x, y, w, h = [int(v) for v in bbox]
        
        padding = 30
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2, y2 = min(image.shape[1], x + w + padding), min(image.shape[0], y + h + padding)
        
        cropped_image = image[y1:y2, x1:x2]
        crop_h, crop_w, _ = cropped_image.shape
        
        scale = self.input_size / max(crop_h, crop_w)
        new_w, new_h = int(crop_w * scale), int(crop_h * scale)
        pad_x, pad_y = (self.input_size - new_w) // 2, (self.input_size - new_h) // 2

        resized_image = cv2.resize(cropped_image, (new_w, new_h))
        padded_image = np.full((self.input_size, self.input_size, 3), 128, dtype=np.uint8)
        padded_image[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized_image

        keypoints = np.array(largest_person_ann['keypoints']).reshape(-1, 3)
        
        # --- Generate Ground Truth Heatmaps ---
        heatmaps = np.zeros((keypoints.shape[0], self.heatmap_size, self.heatmap_size), dtype=np.float32)
        
        for i in range(keypoints.shape[0]):
            if keypoints[i, 2] > 0: # If keypoint is visible
                # Transform keypoint to padded image space
                kp_x = ((keypoints[i, 0] - x1) * scale + pad_x)
                kp_y = ((keypoints[i, 1] - y1) * scale + pad_y)
                
                # Transform to heatmap space
                heatmap_x = int(kp_x / self.stride)
                heatmap_y = int(kp_y / self.stride)

                # Generate Gaussian
                if 0 <= heatmap_x < self.heatmap_size and 0 <= heatmap_y < self.heatmap_size:
                    # Create a grid of coordinates
                    x_grid, y_grid = np.meshgrid(np.arange(self.heatmap_size), np.arange(self.heatmap_size))
                    # Gaussian centered at the keypoint
                    sigma = 1.0
                    g = np.exp(-((x_grid - heatmap_x)**2 + (y_grid - heatmap_y)**2) / (2 * sigma**2))
                    heatmaps[i] = g

        sample = {
            'image': padded_image, 
            'heatmaps': heatmaps,
            'meta': {
                'img_id': img_id, 'ann_id': largest_person_ann['id'],
                'crop_box': [x1, y1, x2, y2], 'scale': scale, 'pad': [pad_x, pad_y]
            }
        }
        
        if self.transform:
            sample['image'] = self.transform(sample['image'])
            
        return sample

# ==============================================================================
# 3. TRAINING AND EVALUATION
# ==============================================================================

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    criterion = nn.MSELoss()

    for i, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        heatmaps = batch['heatmaps'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, heatmaps)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if (i + 1) % 100 == 0:
            print(f"  Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return running_loss / len(dataloader)

def get_coords_from_heatmaps(heatmaps, stride):
    """
    Decodes heatmaps to get (x, y) coordinates and confidence scores.
    Uses a quarter-pixel offset refinement for better accuracy.
    """
    num_keypoints, h, w = heatmaps.shape
    coords = np.zeros((num_keypoints, 2), dtype=np.float32)
    confidences = np.zeros((num_keypoints,), dtype=np.float32)
    
    for i in range(num_keypoints):
        heatmap = heatmaps[i]
        max_val = np.max(heatmap)
        confidences[i] = max_val
        
        # Find the coordinates of the maximum value
        y, x = np.unravel_index(np.argmax(heatmap), (h, w))
        
        # --- FIX: Add quarter-pixel offset for refinement ---
        # Check the values of the neighbors of the max pixel
        if 1 < x < w - 1 and 1 < y < h - 1:
            diff_x = heatmap[y, x+1] - heatmap[y, x-1]
            diff_y = heatmap[y+1, x] - heatmap[y-1, x]
            # Shift the coordinate by 0.25 pixels in the direction of the gradient
            x += 0.25 * np.sign(diff_x)
            y += 0.25 * np.sign(diff_y)
        # --- END FIX ---
        
        coords[i, 0] = x * stride
        coords[i, 1] = y * stride
        
    return coords, confidences

def evaluate(model, dataloader, device, coco_gt):
    model.eval()
    results = []
    stride = dataloader.dataset.stride
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            metas = batch['meta']
            
            outputs = model(images)
            outputs = outputs.cpu().numpy()
            
            for i in range(outputs.shape[0]):
                pred_heatmaps = outputs[i]
                
                # --- FIX: Get both coordinates and confidence scores ---
                pred_coords_padded, pred_confidences = get_coords_from_heatmaps(pred_heatmaps, stride)
                
                scale = metas['scale'][i].item()
                pad_x = metas['pad'][0][i].item()
                pad_y = metas['pad'][1][i].item()
                crop_x1 = metas['crop_box'][0][i].item()
                crop_y1 = metas['crop_box'][1][i].item()

                pred_coords_original = np.zeros_like(pred_coords_padded)
                pred_coords_original[:, 0] = (pred_coords_padded[:, 0] - pad_x) / scale + crop_x1
                pred_coords_original[:, 1] = (pred_coords_padded[:, 1] - pad_y) / scale + crop_y1
                
                # --- FIX: Populate results with confidence scores ---
                keypoints_with_confidence = np.zeros((17, 3))
                keypoints_with_confidence[:, :2] = pred_coords_original
                keypoints_with_confidence[:, 2] = pred_confidences
                
                result = {
                    "image_id": metas['img_id'][i].item(),
                    "category_id": 1,
                    "keypoints": keypoints_with_confidence.flatten().tolist(),
                    "score": float(pred_confidences.mean()) # Use average keypoint confidence as person score
                }
                # --- END FIX ---
                results.append(result)

    if not results:
        return 0.0

    res_file = "temp_results.json"
    with open(res_file, "w") as f:
        json.dump(results, f)
        
    coco_dt = coco_gt.loadRes(res_file)
    coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    ap = coco_eval.stats[0]
    os.remove(res_file)
    
    return ap

# ==============================================================================
# 5. MAIN EXECUTION BLOCK
# ==============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    TRAIN_IMG_DIR, TRAIN_ANN_FILE, VAL_IMG_DIR, VAL_ANN_FILE = get_configuration()
    
    if not os.path.exists(TRAIN_IMG_DIR) or not os.path.exists(VAL_IMG_DIR):
        print("ERROR: Dataset paths not found. Please update paths in main().")
        return

    NUM_EPOCHS = 90 # Heatmap models may benefit from more epochs
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    OUTPUT_FILE_NAME = "heatmap_regression_best"
    AP_RECORD_FILE = f"{OUTPUT_FILE_NAME}.csv"
    
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading training data...")
    train_dataset = CocoHeatmapDataset(root_dir=TRAIN_IMG_DIR, annotation_file=TRAIN_ANN_FILE, transform=data_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    print("Loading validation data...")
    val_dataset = CocoHeatmapDataset(root_dir=VAL_IMG_DIR, annotation_file=VAL_ANN_FILE, transform=data_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = HeatmapPoseModel(num_keypoints=17).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_ap = 0.0
    coco_gt = COCO(VAL_ANN_FILE)

    if AP_RECORD_FILE in os.listdir():
        raise ValueError(f"{AP_RECORD_FILE} already present in current working directory")

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        avg_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1} Average Training Loss: {avg_loss:.4f}")
        
        print("Evaluating on validation set...")
        current_ap = evaluate(model, val_loader, device, coco_gt)
        print(f"Epoch {epoch+1} Validation AP: {current_ap:.4f}")

        with open(AP_RECORD_FILE, "a") as f:
            f.write(current_ap + "\n")
        
        if current_ap > best_ap:
            best_ap = current_ap
            print(f"New best model found! Saving to ")
            torch.save(model.state_dict(), f"{OUTPUT_FILE_NAME}.pth")

    print(f"\n--- Training Finished --- Best Validation AP: {best_ap:.4f}")


if __name__ == '__main__':
    main()
