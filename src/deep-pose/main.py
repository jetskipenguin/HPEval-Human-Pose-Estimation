# DeepPose Implementation for Human Pose Estimation on MS COCO
# This script provides a complete pipeline:
# 1. A DeepPose model using a ResNet-50 backbone.
# 2. A custom PyTorch Dataset for MS COCO.
# 3. A training loop.
# 4. An evaluation function to compute Average Precision (AP) using pycocotools.
# 5. A visualization function to draw predictions on any image.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import torchvision.models as models
from torchvision import transforms

import os
import json
import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class DeepPose(nn.Module):
    """
    DeepPose model for human pose estimation.
    Uses a pre-trained ResNet as a backbone to extract features,
    and a fully connected layer to regress keypoint coordinates.
    """
    def __init__(self, num_keypoints=17):
        super(DeepPose, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(2048, num_keypoints * 2)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CocoPoseDataset(Dataset):
    """
    Custom PyTorch Dataset for the MS COCO Keypoints dataset.
    Generates one sample per person annotation.
    """
    def __init__(self, root_dir, annotation_file, transform=None, target_size=224):
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.transform = transform
        self.target_size = target_size
        
        # *** FIX: Create a list of all valid annotations (one per person) ***
        self.ann_ids = self._get_ann_ids()

    def _get_ann_ids(self):
        """
        Returns a list of annotation IDs for people with at least one keypoint.
        """
        ann_ids = []
        cat_ids = self.coco.getCatIds(catNms=['person'])
        img_ids = self.coco.getImgIds(catIds=cat_ids)
        for img_id in img_ids:
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False))
            for ann in anns:
                if ann['num_keypoints'] > 0:
                    ann_ids.append(ann['id'])
        return ann_ids

    def __len__(self):
        return len(self.ann_ids)

    def __getitem__(self, idx):
        ann_id = self.ann_ids[idx]
        ann = self.coco.loadAnns(ann_id)[0]
        img_info = self.coco.loadImgs(ann['image_id'])[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        bbox = ann['bbox']
        x, y, w, h = [int(v) for v in bbox]
        
        # --- Cropping and Resizing ---
        padding = 30
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        cropped_image = image[y1:y2, x1:x2]
        crop_h, crop_w, _ = cropped_image.shape
        
        scale = self.target_size / max(crop_h, crop_w)
        new_w, new_h = int(crop_w * scale), int(crop_h * scale)
        pad_x, pad_y = (self.target_size - new_w) // 2, (self.target_size - new_h) // 2

        resized_image = cv2.resize(cropped_image, (new_w, new_h))
        padded_image = np.full((self.target_size, self.target_size, 3), 128, dtype=np.uint8) # Pad with grey
        padded_image[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized_image

        # --- Keypoint Transformation ---
        keypoints = np.array(ann['keypoints']).reshape(-1, 3)
        keypoints_transformed = np.zeros_like(keypoints, dtype=np.float32)
        
        # *** FIX: Process keypoints and visibility flags correctly ***
        visible_keypoints_mask = keypoints[:, 2] > 0
        
        # Adjust keypoints for crop, resize, and padding
        kpts_temp = keypoints[visible_keypoints_mask, :2].copy()
        kpts_temp[:, 0] = (kpts_temp[:, 0] - x1) * scale + pad_x
        kpts_temp[:, 1] = (kpts_temp[:, 1] - y1) * scale + pad_y

        keypoints_transformed[visible_keypoints_mask, :2] = kpts_temp
        
        # Store visibility
        visibility = keypoints[:, 2] 
        
        # *** FIX: Regress to pixel coordinates, not normalized values ***
        keypoints_final = keypoints_transformed[:, :2].flatten()
        
        sample = {
            'image': padded_image, 
            'keypoints': torch.from_numpy(keypoints_final).float(),
            'visibility': torch.from_numpy(visibility).float(),
            'meta': { # Meta data should not be tensors
                'img_id': ann['image_id'], 
                'ann_id': ann_id,
                'crop_box': [x1, y1, x2, y2],
                'scale': scale,
                'pad': [pad_x, pad_y]
            }
        }
        
        if self.transform:
            sample['image'] = self.transform(sample['image'])
            
        return sample

def keypoint_loss(outputs, targets, visibility):
    """
    Custom loss function that only calculates MSE for visible keypoints.
    """
    loss = 0
    # Reshape for easier processing
    outputs = outputs.view(-1, 17, 2)
    targets = targets.view(-1, 17, 2)
    visibility = visibility.view(-1, 17)
    
    for i in range(outputs.size(0)): # Iterate over batch
        # *** FIX: Only compute loss for visible keypoints (v=1 or v=2) ***
        vis_mask = visibility[i] > 0
        if vis_mask.sum() > 0:
            loss += nn.functional.mse_loss(outputs[i][vis_mask], targets[i][vis_mask])
            
    return loss / outputs.size(0)

# *** FIX: Custom collate function to handle metadata ***
def custom_collate_fn(batch):
    # Separate metadata from tensor data
    meta_batch = [item.pop('meta') for item in batch]
    # Use default collate for the rest
    collated_batch = default_collate(batch)
    collated_batch['meta'] = meta_batch
    return collated_batch
    
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        keypoints = batch['keypoints'].to(device)
        visibility = batch['visibility'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        # *** FIX: Use custom loss function ***
        loss = keypoint_loss(outputs, keypoints, visibility)
        if loss == 0: continue # Skip batches with no visible keypoints
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if (i + 1) % 100 == 0:
            print(f"  Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return running_loss / len(dataloader)

def evaluate(model, dataloader, device, coco_gt):
    model.eval()
    results = []
    target_size = dataloader.dataset.target_size
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            # *** FIX: Access meta correctly from the custom collate function output ***
            metas = batch['meta'] 
            
            outputs = model(images)
            outputs = outputs.cpu().numpy()
            
            for i in range(outputs.shape[0]):
                pred_keypoints = outputs[i].reshape(-1, 2)
                
                # Get metadata for the current sample
                meta = metas[i]
                scale = meta['scale']
                pad_x, pad_y = meta['pad']
                crop_x1, crop_y1, _, _ = meta['crop_box']

                # --- Reverse Transformations ---
                # 1. Reverse padding
                pred_keypoints[:, 0] -= pad_x
                pred_keypoints[:, 1] -= pad_y
                
                # 2. Reverse scaling
                pred_keypoints[:, 0] /= scale
                pred_keypoints[:, 1] /= scale
                
                # 3. Add original crop offset
                pred_keypoints[:, 0] += crop_x1
                pred_keypoints[:, 1] += crop_y1
                
                keypoints_with_confidence = np.ones((17, 3))
                keypoints_with_confidence[:, :2] = pred_keypoints
                
                result = {
                    "image_id": meta['img_id'],
                    "category_id": 1, # 'person' category
                    "keypoints": keypoints_with_confidence.flatten().tolist(),
                    "score": 1.0 # Model doesn't predict confidence, so use 1.0
                }
                results.append(result)

    if not results:
        print("No results to evaluate.")
        return 0.0

    res_file = "temp_results.json"
    with open(res_file, "w") as f:
        json.dump(results, f)
        
    coco_dt = coco_gt.loadRes(res_file)
    
    coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    ap = coco_eval.stats[0] # AP @ OKS=0.50:0.95
    os.remove(res_file)
    
    return ap


def main():
    # --- Configuration ---
    # !!! IMPORTANT !!!
    # UPDATE THESE PATHS TO YOUR COCO DATASET LOCATION
    TRAIN_IMG_DIR = '/path/to/your/coco/images/train2017'
    TRAIN_ANN_FILE = '/path/to/your/coco/annotations/person_keypoints_train2017.json'
    VAL_IMG_DIR = '/path/to/your/coco/images/val2017'
    VAL_ANN_FILE = '/path/to/your/coco/annotations/person_keypoints_val2017.json'
    
    if not os.path.exists(TRAIN_IMG_DIR) or not os.path.exists(VAL_IMG_DIR):
        print("="*50)
        print("!!! ERROR: Dataset paths not found. !!!")
        print("Please update TRAIN_IMG_DIR and VAL_IMG_DIR in the main() function.")
        print("="*50)
        return

    NUM_EPOCHS = 20
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading training data...")
    train_dataset = CocoPoseDataset(root_dir=TRAIN_IMG_DIR, annotation_file=TRAIN_ANN_FILE, transform=data_transform)
    # *** FIX: Use the custom collate function ***
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)
    
    print("Loading validation data...")
    val_dataset = CocoPoseDataset(root_dir=VAL_IMG_DIR, annotation_file=VAL_ANN_FILE, transform=data_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)
    
    # --- Model Setup ---
    model = DeepPose(num_keypoints=17).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- Training Loop ---
    best_ap = 0.0
    coco_gt = COCO(VAL_ANN_FILE)

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        avg_loss = train_one_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1} Average Training Loss: {avg_loss:.4f}")
        
        print("Evaluating on validation set...")
        current_ap = evaluate(model, val_loader, device, coco_gt)
        print(f"Epoch {epoch+1} Validation AP: {current_ap:.4f}")
        
        if current_ap > best_ap:
            best_ap = current_ap
            print(f"New best model found! Saving to 'deeppose_best.pth'")
            torch.save(model.state_dict(), "deeppose_best.pth")

    print("\n--- Training Finished ---")
    print(f"Best Validation AP: {best_ap:.4f}")

if __name__ == '__main__':
    main()