# UNet_Polyp_Segmentation
## Description
This project focuses on polyp segmentation in medical images using the UNet model. It is part of a Kaggle competition, with the goal of accurately segmenting polyps in endoscopic images to achieve a minimum score of 0.7 on the Kaggle leaderboard.

## Setup_Instructions
### Install_Dependencies
```bash
pip install torch torchvision numpy opencv-python matplotlib wandb
```

### Clone_Repository
```bash
git clone 
cd 
```

### Configure_WandB
```bash
wandb login
```

## Project_Structure
- unet_model.py: Contains the architecture of the UNet model
- train.py: Script to train the model on the polyp segmentation dataset
- infer.py: Script to perform segmentation on test images using the trained model
- best_model_checkpoint.pth: The saved model checkpoint after training
- README.md: This file, providing instructions on how to use the project

## Usage
### Train_Model
```bash
python3 train.py
```

### Run_Inference
```bash
python3 infer.py --image_path image.jpeg
```

### Submit_to_Kaggle
1. Generate predictions on test set
2. Format predictions per competition guidelines
3. Upload predictions and model checkpoint
4. Submit for evaluation

## Links
Kaggle_Competition: [Kaggle Polyp Segmentation](https://www.kaggle.com/c/bkai-igh-neopolyp/overview)

