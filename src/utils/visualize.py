import torch
from torchvision import transforms

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# 4. VISUALIZATION
# ==============================================================================

SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

def visualize(model, image_path, device, target_size=224):
    model.eval()
    
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Error: Could not load image at {image_path}")
        return
        
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_h, original_w, _ = image_rgb.shape

    scale = target_size / max(original_h, original_w)
    new_w, new_h = int(original_w * scale), int(original_h * scale)
    pad_x, pad_y = (target_size - new_w) // 2, (target_size - new_h) // 2

    resized_image = cv2.resize(image_rgb, (new_w, new_h))
    padded_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    padded_image[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized_image

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(padded_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
    
    pred_keypoints = outputs.cpu().numpy()[0].reshape(-1, 2)
    
    pred_keypoints[:, 0] = (pred_keypoints[:, 0] * target_size - pad_x) / scale
    pred_keypoints[:, 1] = (pred_keypoints[:, 1] * target_size - pad_y) / scale
    
    vis_image = image_bgr.copy()
    
    for i, (p1_idx, p2_idx) in enumerate(SKELETON):
        pt1 = tuple(pred_keypoints[p1_idx - 1].astype(int))
        pt2 = tuple(pred_keypoints[p2_idx - 1].astype(int))
        color = COLORS[i]
        cv2.line(vis_image, pt1, pt2, color, 2)
        
    for i, (x, y) in enumerate(pred_keypoints):
        cv2.circle(vis_image, (int(x), int(y)), 5, (0, 0, 255), -1)
        
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Prediction for {os.path.basename(image_path)}")
    plt.axis('off')
    plt.show()