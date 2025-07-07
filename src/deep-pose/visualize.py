import torch
from torchvision import transforms

import cv2
import numpy as np
import argparse
import os
from .main import DeepPose

def visualize_pose(image, keypoints, output_path="output_skeleton.jpg"):
    """
    Draws keypoints and skeleton on an image and saves it.
    """
    # COCO keypoint connections
    skeleton = [
        (16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12), (7, 13),
        (6, 7), (6, 8), (7, 9), (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),
        (2, 4), (3, 5), (4, 6), (5, 7)
    ]
    
    vis_image = image.copy()
    
    # Draw keypoints
    for i in range(keypoints.shape[0]):
        x, y = keypoints[i]
        cv2.circle(vis_image, (int(x), int(y)), 5, (0, 255, 0), -1)

    # Draw skeleton
    for connection in skeleton:
        kp1_idx, kp2_idx = connection[0] - 1, connection[1] - 1
        pt1 = (int(keypoints[kp1_idx, 0]), int(keypoints[kp1_idx, 1]))
        pt2 = (int(keypoints[kp2_idx, 0]), int(keypoints[kp2_idx, 1]))
        cv2.line(vis_image, pt1, pt2, (255, 0, 0), 2)
        
    cv2.imwrite(output_path, vis_image)
    print(f"Model inference visualization image saved to {output_path}")


def predict_on_image(model, image_path, device, target_size=224):
    """
    Performs inference on a single image.
    Since we don't have a bounding box, we process the whole image.
    """
    model.eval()
    
    # Image preprocessing
    image = cv2.imread(image_path)
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    h, w, _ = original_image.shape
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    pad_x, pad_y = (target_size - new_w) // 2, (target_size - new_h) // 2
    
    resized_image = cv2.resize(original_image, (new_w, new_h))
    padded_image = np.full((target_size, target_size, 3), 128, dtype=np.uint8)
    padded_image[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_image
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(padded_image).unsqueeze(0).to(device)
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        
    pred_keypoints = outputs.cpu().numpy()[0].reshape(-1, 2)
    
    # Reverse the transformations to map keypoints back to original image
    pred_keypoints[:, 0] = (pred_keypoints[:, 0] - pad_x) / scale
    pred_keypoints[:, 1] = (pred_keypoints[:, 1] - pad_y) / scale
    
    # Use the original BGR image for OpenCV drawing
    visualize_pose(image, pred_keypoints)

def main():
    parser = argparse.ArgumentParser(description="DeepPose Training and Prediction")
    parser.add_argument('--image_path', type=str, default='test_image.jpg',
                        help="Path to the image for prediction")
    parser.add_argument('--model_path', type=str, default='deeppose_best.pth',
                        help="Path to the trained model file")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Starting in Prediction Mode...")
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
    elif not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
    else:
        # --- Model Setup ---
        model = DeepPose(num_keypoints=17).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        
        # --- Run Prediction ---
        predict_on_image(model, args.image_path, device)
