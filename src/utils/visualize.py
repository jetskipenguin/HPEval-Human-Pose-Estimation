import torch
from torchvision import transforms

import os
import cv2
import matplotlib.pyplot as plt

# Define the connections for the COCO skeleton
SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
    [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7]
]

# Define colors for the skeleton
COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85]
]

def visualize(model, image_path, device):
    """
    Runs the model on a single image and displays the results.
    """
    model.eval()
    
    # 1. Load and transform the image
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Error: Could not load image at {image_path}")
        return
        
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_h, original_w, _ = image_rgb.shape

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image_rgb).unsqueeze(0).to(device)

    # 2. Get model prediction
    with torch.no_grad():
        outputs = model(image_tensor)
    
    pred_keypoints = outputs.cpu().numpy()[0].reshape(-1, 2)
    
    # 3. De-normalize keypoints to original image dimensions
    pred_keypoints[:, 0] *= original_w
    pred_keypoints[:, 1] *= original_h
    
    # 4. Draw skeleton and keypoints on the image
    vis_image = image_bgr.copy()
    
    # Draw skeleton
    for i, (p1_idx, p2_idx) in enumerate(SKELETON):
        # COCO keypoint indices are 1-based, array is 0-based
        pt1 = tuple(pred_keypoints[p1_idx - 1].astype(int))
        pt2 = tuple(pred_keypoints[p2_idx - 1].astype(int))
        color = COLORS[i]
        cv2.line(vis_image, pt1, pt2, color, 2)
        
    # Draw keypoints
    for i, (x, y) in enumerate(pred_keypoints):
        cv2.circle(vis_image, (int(x), int(y)), 5, (0, 0, 255), -1)
        
    # 5. Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Prediction for {os.path.basename(image_path)}")
    plt.axis('off')
    plt.show()