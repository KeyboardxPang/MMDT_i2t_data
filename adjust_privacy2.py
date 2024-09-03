import pandas as pd
import json
from glob import glob
from natsort import natsorted
from datasets import Dataset, Image
import os

# key in the json file
# common key:
# 1. "task": "street_view" or "selfies"
# 2. "image": the path to the image
# 3. "id": name of the file

# key in the street_view:
# 4. "type_street_view"
# 5. "country"
# 6. "state_province"
# 7. "city"
# 8. "latitude"
# 9. "longitude
# 10. "zipcode"

# key in the selfies:
# 11. "ethnicity"
# 12. "label_selfies"
# 13. "type_selfies"

# split of the dataset
# 1. street view
# 2. selfies

ethnicity_list = ["caucasians", "hispanics"]
root_folder = "/Users/yuanlingzhi/Desktop/change_data/data/image-to-text/privacy/Selfies/pii"
task = "selfies"

data_list = []
for ethnicity in ethnicity_list:
    folder_path = os.path.join(root_folder, ethnicity)
    dir_list = os.listdir(folder_path)
    dir_list = natsorted(dir_list)
    for di in dir_list:
        label_selfies = di.split("/")[-1]
        image_paths = glob(os.path.join(folder_path, di) + "/*.jpg")
        image_paths = natsorted(image_paths)
        for image_path in image_paths:
            type_selfies = image_path.split("/")[-1].split("_")[0]
            data = {}
            data["task"] = task
            data["image"] = image_path
            data["id"] = image_path.split("/")[-1].split(".")[0]
            data["type_street_view"] = ""
            data["country"] = ""
            data["state_province"] = ""
            data["city"] = ""
            data["latitude"] = 1.11111
            data["longitude"] = 1.11111
            data["zipcode"] = ""
            data["ethnicity"] = ethnicity
            data["label_selfies"] = label_selfies
            data["type_selfies"] = type_selfies
            data_list.append(data)

# save the data to a .json file
file_path = "/Users/yuanlingzhi/Desktop/change_data/test_data/privacy/selfies.json"
with open(file_path, 'w') as f:
    json.dump(data_list, f, indent=4)

dataset = Dataset.from_json(file_path)
dataset = dataset.cast_column("image", Image())

# push the dataset to huggingface repository
dataset.push_to_hub("YuanXiaopang/test_mmdt", config_name="privacy", split="selfies")