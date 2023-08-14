import csv
import json
import torch
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
    num_ROI = 6
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model_name = "ViT-L/14@336px"
    out_embd_path = "/rds/user/kn413/hpc-work/Visual-Question-Answering/data/ok-vqa/pre-extracted_features/ROI/ROI6_train2014_extracted_with_vinvl_large.pkl"
    out_img_path = "/rds/user/kn413/hpc-work/Visual-Question-Answering/src/tools/Testing/ROI_images"

    vinvl_features_path = "/rds/user/kn413/hpc-work/Visual-Question-Answering/data/ok-vqa/pre-extracted_features/vinvl_output/vinvl_okvqa_trainset_full/inference/vinvl_vg_x152c4/predictions.tsv"
    # vinvl_features_path = "/rds/user/kn413/hpc-work/Visual-Question-Answering/data/ok-vqa/pre-extracted_features/vinvl_output/vinvl_okvqa_testset_full/inference/vinvl_vg_x152c4/predictions.tsv"
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

    predictions = read_tsv_file(vinvl_features_path)

    ROI_embeddings = {}
    count = 0
    total_cont = 0
    dupli = 0

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
        
        prediction = predictions[img_key]
        prediction = json.loads(prediction)

        selected_objects = []
        objects = []

        for obj in prediction['objects']:

            xmin, ymin, xmax, ymax = obj['rect']
            obj_area = (ymax - ymin) * (xmax - xmin)
            objects.append((obj_area, obj))

            # #if object is part of the question keep it first! VIDI GDE OVO DA STAVIS A DA PRIORITIZUJES
            # if obj['class'].lower().strip() in data_item["question"].lower():
            #     selected_objects.append(obj)
            #     print(data_item["question"])
            #     print(obj["class"])
            #     count = count + 1

        objects = sorted(objects, key=lambda x: x[0], reverse=True)

        # samo ako je u question
        for obj_area, obj in objects:
            xmin, ymin, xmax, ymax = obj['rect']
            if len(selected_objects) >= num_ROI:
                break
            else:
                valid = True

                # Remove duplications
                for existing_obj in selected_objects:
                    if existing_obj['class'] == obj['class']:
                        e_xmin, e_ymin, e_xmax, e_ymax = existing_obj['rect']
                        if xmin >= e_xmin and ymin >= e_ymin and xmax <= e_xmax and ymax <= e_ymax:
                            # this object is contained in an existing object with the same class name
                            valid = False
                if valid:
                    # if object is part of the question keep it first! VIDI GDE OVO DA STAVIS A DA PRIORITIZUJES
                    if is_word_in_string(obj['class'].lower().strip(), data_item["question"].lower()):
                        selected_objects.append(obj)
                        print(data_item["question"])
                        print(obj["class"])
                        print(data_item["image_id"])

                        # xmin, ymin, xmax, ymax = obj['rect']
                        # xmin, ymin, xmax, ymax = int(round(xmin)), int(round(ymin)), int(round(xmax)), int(round(ymax))
                        # cropped_image_ = image[ymin:ymax, xmin:xmax]
                        # new_id = f"{out_img_path}/{data_item['image_id']}_{obj['class']}_{xmin}_{ymin}_{xmax}_{ymax}.jpg"
                        # io.imsave(new_id, cropped_image_)

                        count = count + 1

        for obj_area, obj in objects:
            xmin, ymin, xmax, ymax = obj['rect']
            if len(selected_objects) >= num_ROI:
                break
            else:
                valid = True
                # Remove duplications
                for existing_obj in selected_objects:
                    if existing_obj['class'] == obj['class']:
                        e_xmin, e_ymin, e_xmax, e_ymax = existing_obj['rect']
                        if xmin >= e_xmin and ymin >= e_ymin and xmax <= e_xmax and ymax <= e_ymax:
                            # this object is contained in an existing object with the same class name
                            valid = False
                # print(obj['conf'], obj['class'])
                if obj['conf'] < 0.5:
                    # print(obj['class'], obj['conf'])
                    valid = False
                if valid:
                    selected_objects.append(obj)
        
        # ponovi ali bez uslova za confidence
        if len(selected_objects) < num_ROI:
            for obj_area, obj in objects:
                xmin, ymin, xmax, ymax = obj['rect']
                if len(selected_objects) >= num_ROI:
                    break
                else:
                    valid = True
                    # Remove duplications
                    for existing_obj in selected_objects:
                        if existing_obj['class'] == obj['class']:
                            e_xmin, e_ymin, e_xmax, e_ymax = existing_obj['rect']
                            if xmin >= e_xmin and ymin >= e_ymin and xmax <= e_xmax and ymax <= e_ymax:
                                # this object is contained in an existing object with the same class name
                                valid = False
                    if valid:
                        selected_objects.append(obj)

        # ponovi bez skroz istih
        if len(selected_objects) < num_ROI:
            for obj_area, obj in objects:
                xmin, ymin, xmax, ymax = obj['rect']
                if len(selected_objects) >= num_ROI:
                    break
                else:
                    valid = True
                    # Remove duplications
                    for existing_obj in selected_objects:
                        if existing_obj['class'] == obj['class']:
                            e_xmin, e_ymin, e_xmax, e_ymax = existing_obj['rect']
                            if xmin == e_xmin and ymin == e_ymin and xmax == e_xmax and ymax == e_ymax:
                                # this object is contained in an existing object with the same class name
                                valid = False
                    if valid:
                        selected_objects.append(obj)

        # ponovi samo da popunis
        if len(selected_objects) < num_ROI:
            for obj_area, obj in objects:
                xmin, ymin, xmax, ymax = obj['rect']
                if len(selected_objects) >= num_ROI:
                    break
                else:
                    selected_objects.append(obj)
                    dupli = dupli + 1

        # add image embeddings
    
        whole_image = image_preprocessor(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = model.encode_image(whole_image).cpu().numpy().astype(np.float32)

        if question_id not in ROI_embeddings:
            ROI_embeddings[question_id] = []

        ROI_embeddings[question_id].append(prefix)

        ROIs = []
        # print("Bilo je:", len(objects), "Izabrano: ", len(selected_objects))
        for obj in selected_objects:
            xmin, ymin, xmax, ymax = obj['rect']
            xmin, ymin, xmax, ymax = int(round(xmin)), int(round(ymin)), int(round(xmax)), int(round(ymax))
            cropped_image = image[ymin:ymax, xmin:xmax]
            # new_id = f"{out_img_path}/{obj['class']}_{xmin}_{ymin}_{xmax}_{ymax}.jpg"
            # io.imsave(new_id, cropped_image)
            # print(obj['rect'], obj['class'], obj['conf'])
            cropped_image = image_preprocessor(Image.fromarray(cropped_image)).unsqueeze(0).to(device)
        
            with torch.no_grad():
                prefix = model.encode_image(cropped_image).cpu().numpy().astype(np.float32)

            ROI_embeddings[question_id].append(prefix)
            total_cont = total_cont + 1

        if (i + 1) % 10000 == 0:
            with open(out_embd_path, "wb") as f:
                pickle.dump(ROI_embeddings, f)

    with open(out_embd_path, "wb") as f:
        pickle.dump(ROI_embeddings, f)

    print("Done")
    print("%0d embeddings saved " % len(ROI_embeddings))
    print("Precents of ROIs are part of the question", count/total_cont*100)
    print("Precents of ROIs doubled", dupli/total_cont*100)

if __name__ == "__main__":
    main()