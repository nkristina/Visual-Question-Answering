import csv
import json
import torch
import matplotlib.pyplot as plt
from PIL import Image
import skimage.io as io
import numpy as np
from tqdm import tqdm

from pathlib import Path
import clip

def main():

    data_dir = Path("../data")
    okvqa_data_dir = data_dir / "ok-vqa"
    num_ROI = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model_name = "ViT-L/14@336px"
    out_embd_path = "/rds/user/kn413/hpc-work/Visual-Question-Answering/data/ok-vqa/pre-extracted_features/ROI/ROI4_val2014_extracted_with_vinvl_large.pkl"
    out_img_path = "/rds/user/kn413/hpc-work/Visual-Question-Answering/src/tools/Testing/ROI_images"

    vinvl_large_features_path = "../data/ok-vqa/pre-extracted_features/vinvl_output/vinvl_large_okvqa_testset/inference/vinvl_large/predictions.tsv"
    vinvl_features_path = "/rds/user/kn413/hpc-work/Visual-Question-Answering/data/ok-vqa/pre-extracted_features/vinvl_output/vinvl_okvqa_testset_full/inference/vinvl_vg_x152c4/predictions.tsv"
    subtype = "val2014"

    with open(
        okvqa_data_dir / f"OpenEnded_mscoco_{subtype}_questions.json", "r"
    ) as f:
        data = json.load(f)

    questions = data["questions"]
    print("%0d questions loaded from json " % len(questions))

    print("uso")
    model, image_preprocessor = clip.load(clip_model_name, device=device)
    clip_model_name = clip_model_name.replace('/', '_')

    tsv_file = open(vinvl_features_path)
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    csv.field_size_limit(100000000)

    ROI_embeddings = {}
    i = 0
    for row in tqdm(read_tsv):
        img_id, prediction = row
        prediction = json.loads(prediction)
        # ucitaj sliku
        img_path = okvqa_data_dir / f"{subtype}/COCO_{subtype}_{int(img_id):012d}.jpg"
        image = io.imread(img_path)
        selected_objects = []
        objects = []

        for obj in prediction['objects']:

            xmin, ymin, xmax, ymax = obj['rect']
            obj_area = (ymax - ymin) * (xmax - xmin)
            objects.append((obj_area, obj))

            # if object is part of the question keep it first! VIDI GDE OVO DA STAVIS A DA PRIORITIZUJES
            # if obj['class'].lower().strip() in item.question.lower():
            #     selected_objects.append(obj)

        objects = sorted(objects, key=lambda x: x[0], reverse=True)

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

        # add image embeddings
    
        whole_image = image_preprocessor(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = model.encode_image(whole_image).cpu().numpy().astype(np.float32)

        if img_id not in ROI_embeddings:
            ROI_embeddings[img_id] = []

        ROI_embeddings[img_id].append(prefix)

        ROIs = []
        print("Bilo je:", len(objects), "Izabrano: ", len(selected_objects))
        for obj in selected_objects:
            xmin, ymin, xmax, ymax = obj['rect']
            xmin, ymin, xmax, ymax = int(round(xmin)), int(round(ymin)), int(round(xmax)), int(round(ymax))
            cropped_image = image[ymin:ymax, xmin:xmax]
            # new_id = f"{out_img_path}/{obj['class']}_{xmin}_{ymin}_{xmax}_{ymax}.jpg"
            # io.imsave(new_id, cropped_image)
            print(obj['rect'], obj['class'], obj['conf'])
            cropped_image = image_preprocessor(Image.fromarray(cropped_image)).unsqueeze(0).to(device)
        
            with torch.no_grad():
                prefix = model.encode_image(cropped_image).cpu().numpy().astype(np.float32)

            ROI_embeddings[img_id].append(prefix)

        if (i + 1) % 10000 == 0:
            if not os.path.exists(out_embd_path.parent):
                os.makedirs(out_embd_path.parent)
            with open(out_embd_path, "wb") as f:
                pickle.dump(ROI_embeddings, f)
        i = i + 1

    if not os.path.exists(out_embd_path.parent):
        os.makedirs(out_embd_path.parent)
    with open(out_embd_path, "wb") as f:
        pickle.dump(ROI_embeddings, f)

    print("Done")
    print("%0d embeddings saved " % len(ROI_embeddings))

if __name__ == "__main__":
    main()