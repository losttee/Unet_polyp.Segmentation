# UNet_Polyp_Segmentation
## Description
This project is designed for medical image segmentation, specifically focused on polyp segmentation in endoscopic images. The task is part of a Kaggle competition, where the objective is to accurately predict segmentation masks for polyps, achieving a minimum score of 0.7 on the leaderboard. The model used is based on the UNet++ architecture, which is optimized for high-quality segmentation tasks.

## Setup_Instructions
### Install_Dependencies
Ensure you have Python installed, then run the following command to install the necessary libraries
```bash
pip install torch torchvision numpy opencv-python matplotlib wandb albumentations segmentation-models-pytorch  
```

### Clone_Repository
```bash
git clone 
cd 
```

### Configure_WandB
Download the repository to your local machine:
```bash
git clone Unet_polyp.Segmentation
cd Unet_polyp.Segmentation 
```

## Project_Structure
- unet_model.py: Contains the implementation of the UNet++ model architecture.
- train.py: Script to train the UNet++ model using the polyp segmentation dataset.
- infer.py: Script to perform inference on test images using the trained model checkpoint.
- best_model_checkpoint.pth: The best model checkpoint saved during training.
- output.csv: File containing RLE-encoded predictions for test images, ready for Kaggle submission.
- README.md: Documentation and instructions on using the project.

## Usage
### Train_Model
To train the model on the dataset, run the following command:
```bash
python3 train.py
```
The training script will:
- Load and preprocess the dataset.
- Apply data augmentation techniques.
- Train the UNet++ model while logging the training/validation metrics to WandB.
- Save the best model checkpoint as best_model_checkpoint.pth.

### Run_Inference
After training, use the infer.py script to generate segmentation predictions for a single image:
```bash
python3 infer.py --image_path image.jpeg
```
The segmented output will be saved in the working directory with the same name as the input image.

### Submit_to_Kaggle
1. Generate predictions on test set
2. Format predictions per competition guidelines
3. Upload predictions and model checkpoint
4. Submit for evaluation

## Data Augmentation Techniques
- Rotation: Random rotation of images within the range of -20° to +20°.
- Flipping: Random horizontal flips with a 50% probability.
- Scaling: Random scaling to accommodate variations in object size.
- Brightness Adjustment: Adjusts brightness to simulate different lighting conditions.
- Normalization: Normalizes pixel values to the range [0, 1].

## Links
Kaggle_Competition: [Kaggle Polyp Segmentation](https://www.kaggle.com/c/bkai-igh-neopolyp/overview)

