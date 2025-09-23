"""
Creates the app.py script for deployment.
"""
# 1. In git bash, cd to C:/data/dev/workspace
# 2. To use, do git clone https://huggingface.co/spaces/[YOUR_USERNAME]/[YOUR_SPACE_NAME]
# 3. Copy to contents of foodvision_big directory into the folder created
# 4. cd into the folder
# 5. have git lfs installed and run "git lfs track "*.file_extension""
# 6. run "git add .gitattributes"
# 7. run "git add *"
# 8. run "git -c user.name="Your Name" -c user.email="you@example.com" commit -m "message"" (one time identity)
# 9. run "git push"

# 1. Imports and class names setup
import gradio as gr
import os
import torch

from model import create_effnetb0_model_deployment
from timeit import default_timer as timer
from typing import Tuple, Dict

# Set up class names
with open("class_names.txt", "r") as f:  # reading them in from class_names.txt
    class_names = [food_name.strip() for food_name in f.readlines()]

# 2. Model and transforms preparation

# Create an EffNetB0 model capable of fitting to 101 classes for Food101
effnetb0, effnetb0_transforms = create_effnetb0_model_deployment(
    len(class_names),
    device="cpu"
)

# Load saved weights
effnetb0.load_state_dict(
    torch.load(
        f="pretrained_effnetb0_feature_extractor_food101_20_percent.pth",
        map_location=torch.device("cpu"),  # load to CPU
    )
)


# 3. Predict function

# Create predict function
def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns a prediction with the time taken."""
    # Start the timer
    start_time = timer()

    # Transform the target image and add a batch dimension
    img = effnetb0_transforms(img).unsqueeze(0)

    # Put the model into evaluation mode and turn on inference mode
    effnetb0.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(effnetb0(img), dim=1)

    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)

    # Return the prediction dictionary and prediction time
    return pred_labels_and_probs, pred_time


# Create title, description and article strings
title = "FoodVision Big"
description = "An EfficientNetB0 feature extractor computer vision model to classify images of food into [101 different classes] "
article = "Created by good guy."

# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio interface
demo = gr.Interface(fn=predict,  # mapping function from input to output
                    inputs=gr.Image(type="pil"),  # what are the inputs?
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"),  # what are the outputs?
                             gr.Number(label="Prediction time (s)")],  # our fn has two outputs, therefore we have two outputs
                    # Create examples list from "examples/" directory
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

# Launch the app!
demo.launch()
