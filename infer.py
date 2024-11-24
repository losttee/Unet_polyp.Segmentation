import torch
import cv2
import numpy as np
import argparse
from segmentation_models_pytorch import UnetPlusPlus
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2

# Color dictionary for segmentation mask
color_dict = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0)}

def mask_to_rgb(mask, color_dict):
    output = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for k, v in color_dict.items():
        output[mask == k] = v
    return output

def load_model(checkpoint_path, device):
    model = UnetPlusPlus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=3)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model

def predict_image(image_path, model, transform, device):
    ori_img = cv2.imread(image_path)
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    ori_h, ori_w = ori_img.shape[:2]

    img = cv2.resize(ori_img, (256, 256))
    transformed = transform(image=img)
    input_img = transformed['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        output_mask = model(input_img).squeeze(0).cpu().numpy().transpose(1, 2, 0)
        mask = cv2.resize(output_mask, (ori_w, ori_h))
        mask = np.argmax(mask, axis=2)
        mask_rgb = mask_to_rgb(mask, color_dict)

    return mask_rgb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image")
    args = parser.parse_args()

    checkpoint_path = "colorization_model.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = load_model(checkpoint_path, device)

    # Define transforms
    transform = Compose([
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # Perform inference
    segmented_image = predict_image(args.image_path, model, transform, device)

    # Save the result
    output_path = "segmented_output.png"
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, segmented_image)
    print(f"Segmented image saved at {output_path}")
