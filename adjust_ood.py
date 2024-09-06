import pandas as pd
import json
from glob import glob
from natsort import natsorted
from datasets import Dataset, Image
import os
import pandas as pd

# 7 splits
splits = ["original", "Van_Gogh", "oil_painting", "watercolour_painting", "zoom_blur", "gaussian_noise", "pixelate"]

# each data has 6 keys
# 1. id
# 2. img_id
# 3. question
# 4. answer
# 5. task
# 6. image

tasks = ["attribute", "count", "identification", "spatial"]

image_root_path = "/Users/yuanlingzhi/Desktop/change_data/data/image-to-text/ood/images"
json_root_path = "/Users/yuanlingzhi/Desktop/change_data/data/image-to-text/ood"

for split in splits:
    # get the split data from the 4 json files of the 4 tasks
    data_list = []
    for task in tasks:
        file_path = os.path.join(json_root_path, f"{task}.json")
        with open(file_path, 'r') as f:
            whole_data = json.load(f)
        
        for key, value in whole_data[split].items():
            entry = {
                'id': key,
                'img_id': value['img_id'],
                'question': value['question'],
                'answer': value['answer'],
                'task': task,
                'image': os.path.join(image_root_path + f"/{split}", f"{value['img_id']}.jpg")
            }
            data_list.append(entry)

    # save the data to a .json file
    file_path = f"/Users/yuanlingzhi/Desktop/change_data/MMDT_i2t_data/test_data/ood/{split}.json"
    with open(file_path, 'w') as f:
        json.dump(data_list, f, indent=4)
    
    dataset = Dataset.from_json(file_path)
    dataset = dataset.cast_column("image", Image())

    # push the dataset to huggingface repository
    dataset.push_to_hub("YuanXiaopang/test_mmdt", config_name="ood", split=f"{split}")