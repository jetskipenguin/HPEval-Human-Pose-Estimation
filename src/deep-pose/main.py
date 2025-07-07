# DeepPose Implementation for Human Pose Estimation on MS COCO
# This script provides a complete pipeline:
# 1. A DeepPose model using a ResNet backbone.
# 2. A custom PyTorch Dataset for MS COCO.
# 3. A training loop.
# 4. An evaluation function to compute Average Precision (AP) using pycocotools.
# 5. A visualization function to draw predictions on any image.

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


class DeepPose(nn.Module):
    """
    DeepPose model for human pose estimation.
    Uses a pre-trained ResNet as a backbone to extract features,
    and a fully connected layer to regress keypoint coordinates.
    """
    def __init__(self, num_keypoints=17):
        """
        Initializes the DeepPose model.
        Args:
            num_keypoints (int): The number of keypoints to predict (17 for COCO).
        """
        super(DeepPose, self).__init__()
        
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # We use all layers of ResNet except for the final classification layer (fc)
        # The output of the layer before fc is a 2048-dimensional feature vector.
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Define the regression head
        # It takes the 2048-D feature vector and outputs 2 coordinates (x, y) for each keypoint.
        self.fc = nn.Linear(2048, num_keypoints * 2)

    def forward(self, x):
        """
        Forward pass through the model.
        Args:
            x (torch.Tensor): The input image tensor.
        Returns:
            torch.Tensor: The predicted keypoint coordinates (batch_size, num_keypoints * 2).
        """
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # Flatten the features
        x = self.fc(x)
        return x


class CocoPoseDataset(Dataset):
    """
    Custom PyTorch Dataset for the MS COCO Keypoints dataset.
    """
    def __init__(self, root_dir, annotation_file, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            annotation_file (string): Path to the COCO annotation file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.coco = COCO(annotation_file)
        self.transform = transform
        
        # Filter for images that contain people with keypoints
        self.ids = self._get_image_ids()

    def _get_image_ids(self):
        """
        Returns a list of image IDs that contain at least one person with keypoints.
        """
        ids = []
        cat_ids = self.coco.getCatIds(catNms=['person'])
        for img_id in self.coco.getImgIds():
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False)
            if ann_ids:
                anns = self.coco.loadAnns(ann_ids)
                # Check if any annotation has a non-zero number of keypoints
                if any(ann['num_keypoints'] > 0 for ann in anns):
                    ids.append(img_id)
        return ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding keypoint annotations.
        """
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.coco.getCatIds(catNms=['person']))
        anns = self.coco.loadAnns(ann_ids)
        
        # We'll use the annotation for the largest person in the image
        # This is a simplification for DeepPose which typically handles one person.
        largest_person_ann = None
        max_area = 0
        for ann in anns:
            if ann['num_keypoints'] > 0 and ann['area'] > max_area:
                max_area = ann['area']
                largest_person_ann = ann
        
        if largest_person_ann is None:
            return self.__getitem__((idx + 1) % len(self))
            
        keypoints = np.array(largest_person_ann['keypoints']).reshape(-1, 3)
        
        # Get bounding box to crop the person
        bbox = largest_person_ann['bbox']
        x, y, w, h = [int(v) for v in bbox]
        
        # Crop the image around the person
        padding = 30
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        cropped_image = image[y1:y2, x1:x2]
        
        # Adjust keypoints to be relative to the cropped image
        keypoints[:, 0] -= x1
        keypoints[:, 1] -= y1
        
        # Normalize keypoints
        crop_h, crop_w, _ = cropped_image.shape
        keypoints_normalized = keypoints.copy().astype(np.float32)
        keypoints_normalized[:, 0] /= crop_w
        keypoints_normalized[:, 1] /= crop_h
        
        keypoints_final = keypoints_normalized[:, :2].flatten()
        
        sample = {'image': cropped_image, 'keypoints': keypoints_final, 'meta': {'img_id': img_id, 'ann_id': largest_person_ann['id'], 'crop_box': [x1, y1, x2, y2], 'crop_dims': [crop_w, crop_h]}}
        
        if self.transform:
            sample['image'] = self.transform(sample['image'])
            
        return sample


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        keypoints = batch['keypoints'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, keypoints)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if (i + 1) % 100 == 0:
            print(f"  Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return running_loss / len(dataloader)


def evaluate(model, dataloader, device, coco_gt):
    """
    Evaluates the model on the validation set and computes AP.
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            metas = batch['meta']
            
            outputs = model(images)
            outputs = outputs.cpu().numpy()
            
            for i in range(outputs.shape[0]):
                pred_keypoints = outputs[i].reshape(-1, 2)
                
                crop_w = metas['crop_dims'][0][i].item()
                crop_h = metas['crop_dims'][1][i].item()
                crop_x1 = metas['crop_box'][0][i].item()
                crop_y1 = metas['crop_box'][1][i].item()

                pred_keypoints[:, 0] = pred_keypoints[:, 0] * crop_w + crop_x1
                pred_keypoints[:, 1] = pred_keypoints[:, 1] * crop_h + crop_y1
                
                keypoints_with_confidence = np.ones((17, 3))
                keypoints_with_confidence[:, :2] = pred_keypoints
                
                result = {
                    "image_id": metas['img_id'][i].item(),
                    "category_id": 1,
                    "keypoints": keypoints_with_confidence.flatten().tolist(),
                    "score": 1.0
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
    
    ap = coco_eval.stats[0]
    os.remove(res_file)
    
    return ap


def main():
    # --- Configuration ---
    # !!! IMPORTANT !!!
    # UPDATE THESE PATHS TO YOUR COCO DATASET LOCATION
    TRAIN_IMG_DIR = '/work/vba875/coco/images/train2017'
    TRAIN_ANN_FILE = '/work/vba875/coco/annotations/person_keypoints_train2017.json'
    VAL_IMG_DIR = '/work/vba875/coco/images/val2017'
    VAL_ANN_FILE = '/work/vba875/coco/annotations/person_keypoints_val2017.json'
    
    # Check if paths exist
    if not os.path.exists(TRAIN_IMG_DIR) or not os.path.exists(VAL_IMG_DIR):
        print("="*50)
        print("!!! ERROR: Dataset paths not found. !!!")
        print("Please update TRAIN_IMG_DIR and VAL_IMG_DIR in the main() function.")
        print("="*50)
        return

    # Hyperparameters
    NUM_EPOCHS = 20
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading training data...")
    train_dataset = CocoPoseDataset(root_dir=TRAIN_IMG_DIR, annotation_file=TRAIN_ANN_FILE, transform=data_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    print("Loading validation data...")
    val_dataset = CocoPoseDataset(root_dir=VAL_IMG_DIR, annotation_file=VAL_ANN_FILE, transform=data_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # --- Model Setup ---
    model = DeepPose(num_keypoints=17).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- Training Loop ---
    best_ap = 0.0
    coco_gt = COCO(VAL_ANN_FILE)

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
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

    # --- Visualization ---
    print("\n--- Running Visualization ---")
    # Load the best model for visualization
    vis_model = DeepPose(num_keypoints=17).to(device)
    vis_model.load_state_dict(torch.load("deeppose_best.pth"))

    # !!! IMPORTANT !!!
    # Set this variable to the path of an image you want to test.
    # For example: image_to_visualize = 'path/to/your/image.jpg'
    image_to_visualize = None 

    if image_to_visualize and os.path.exists(image_to_visualize):
        visualize(vis_model, image_to_visualize, device)
    else:
        print("\nTo visualize, please set the 'image_to_visualize' variable")
        print("in the main() function to a valid image path.")


if __name__ == '__main__':
    main()
