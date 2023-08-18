import csv
import json
import torch
import matplotlib.pyplot as plt
from PIL import Image
import skimage.io as io
import numpy as np
from tqdm import tqdm
import os
import pickle

from pathlib import Path
import clip

import re

def is_word_in_string(word, string):
    pattern = r'\b' + re.escape(word) + r'\b'
    return re.search(pattern, string, re.IGNORECASE) is not None

def read_tsv_file(vinvl_features_path):
    tsv_file = open(vinvl_features_path)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    csv.field_size_limit(100000000)

    predictions = {}
    for row in read_tsv:
        img_id, pred = row
        predictions[img_id] = pred

    return predictions

def main():

    data_dir = Path("../data")
    okvqa_data_dir = data_dir / "ok-vqa"
    num_ROI = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model_name = "ViT-L/14@336px"
    out_embd_path = "/rds/user/kn413/hpc-work/Visual-Question-Answering/data/ok-vqa/pre-extracted_features/ROI/slide_ROI4_train2014.pkl"
    out_img_path = "/rds/user/kn413/hpc-work/Visual-Question-Answering/src/tools/Testing/ROI_images"

    subtype = "train2014"

    with open(
        okvqa_data_dir / f"OpenEnded_mscoco_{subtype}_questions.json", "r"
    ) as f:
        data = json.load(f)

    questions = data["questions"]
    print("%0d questions loaded from json " % len(questions))

    print("uso")
    model, image_preprocessor = clip.load(clip_model_name, device=device)
    clip_model_name = clip_model_name.replace('/', '_')

    ROI_embeddings = {}
    count = 0
    total_cont = 0

    for i in tqdm(range(len(questions))):
        data_item = questions[i]

        # print(data_item)
        img_id = str(data_item["image_id"])
        question_id = str(data_item["question_id"])
        img_key = f"{int(img_id):012d}"

        img_path = (
            okvqa_data_dir / f"{subtype}/COCO_{subtype}_{int(img_id):012d}.jpg"
        )
        image = io.imread(img_path)
    
        whole_image = image_preprocessor(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = model.encode_image(whole_image).cpu().numpy().astype(np.float32)

        if question_id not in ROI_embeddings:
            ROI_embeddings[question_id] = []

        ROI_embeddings[question_id].append(prefix)

        ROIs = []

        height, width = image.shape[0], image.shape[1]

        # Calculate half width and half height for cropping
        half_width = width // 2
        half_height = height // 2

        # Crop the images using NumPy slicing
        top_left = image[:half_height, :half_width]
        top_right = image[:half_height, half_width:]
        bottom_left = image[half_height:, :half_width]
        bottom_right = image[half_height:, half_width:]

        # Save the cropped images
        # io.imsave(f"{out_img_path}/{question_id}_top_left.jpg", top_left)
        # io.imsave(f"{out_img_path}/{question_id}_top_right.jpg", top_right)
        # io.imsave(f"{out_img_path}/{question_id}_bottom_left.jpg", bottom_left)
        # io.imsave(f"{out_img_path}/{question_id}_bottom_right.jpg", bottom_right)
 
        cropped_image = image_preprocessor(Image.fromarray(top_left)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prefix = model.encode_image(cropped_image).cpu().numpy().astype(np.float32)

        ROI_embeddings[question_id].append(prefix)

        cropped_image = image_preprocessor(Image.fromarray(top_right)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prefix = model.encode_image(cropped_image).cpu().numpy().astype(np.float32)

        ROI_embeddings[question_id].append(prefix)

        cropped_image = image_preprocessor(Image.fromarray(bottom_left)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prefix = model.encode_image(cropped_image).cpu().numpy().astype(np.float32)

        ROI_embeddings[question_id].append(prefix)

        cropped_image = image_preprocessor(Image.fromarray(bottom_right)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prefix = model.encode_image(cropped_image).cpu().numpy().astype(np.float32)

        ROI_embeddings[question_id].append(prefix)

        if (i + 1) % 10000 == 0:
            with open(out_embd_path, "wb") as f:
                pickle.dump(ROI_embeddings, f)

    with open(out_embd_path, "wb") as f:
        pickle.dump(ROI_embeddings, f)

    print("Done")
    print("%0d embeddings saved " % len(ROI_embeddings))

if __name__ == "__main__":
    main()