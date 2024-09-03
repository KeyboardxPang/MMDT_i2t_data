import pandas as pd
import json
from glob import glob
from datasets import Dataset, Image

def extract_number(file_path):
    return int(file_path.split("/")[-1].split(".")[0])

# read the .json file as a list of dictionary
file_path = "/Users/yuanlingzhi/Desktop/change_data/test_data/adv/attribute.json"
with open(file_path, 'r') as f:
    data = json.load(f)

img_folder_path = "/Users/yuanlingzhi/Desktop/change_data/test_data/adv/attribute"
img_paths = glob(img_folder_path + "/*.png")

# sort the img_paths by the file name
img_paths.sort(key=extract_number)

# add a "img_path" key to each dictionary
for i in range(len(data)):
    data[i]["image"] = img_paths[i]

# save the data to the original .json file
with open(file_path, 'w') as f:
    json.dump(data, f, indent=4)

dataset = Dataset.from_json(file_path)
dataset = dataset.cast_column("image", Image())

# push the dataset to huggingface repository
dataset.push_to_hub("YuanXiaopang/test_mmdt", config_name="adv", split="attribute")


# read the .json file as a list of dictionary
file_path = "/Users/yuanlingzhi/Desktop/change_data/test_data/adv/object.json"
with open(file_path, 'r') as f:
    data = json.load(f)

img_folder_path = "/Users/yuanlingzhi/Desktop/change_data/test_data/adv/object"
img_paths = glob(img_folder_path + "/*.png")

# sort the img_paths by the file name
img_paths.sort(key=extract_number)

# add a "img_path" key to each dictionary
for i in range(len(data)):
    data[i]["image"] = img_paths[i]

# save the data to the original .json file
with open(file_path, 'w') as f:
    json.dump(data, f, indent=4)

dataset = Dataset.from_json(file_path)
dataset = dataset.cast_column("image", Image())

# push the dataset to huggingface repository
dataset.push_to_hub("YuanXiaopang/test_mmdt", config_name="adv", split="object")

# read the .json file as a list of dictionary
file_path = "/Users/yuanlingzhi/Desktop/change_data/test_data/adv/spatial.json"
with open(file_path, 'r') as f:
    data = json.load(f)

img_folder_path = "/Users/yuanlingzhi/Desktop/change_data/test_data/adv/spatial"
img_paths = glob(img_folder_path + "/*.png")

# sort the img_paths by the file name
img_paths.sort(key=extract_number)

# add a "img_path" key to each dictionary
for i in range(len(data)):
    data[i]["image"] = img_paths[i]

# save the data to the original .json file
with open(file_path, 'w') as f:
    json.dump(data, f, indent=4)

dataset = Dataset.from_json(file_path)
dataset = dataset.cast_column("image", Image())

# push the dataset to huggingface repository
dataset.push_to_hub("YuanXiaopang/test_mmdt", config_name="adv", split="spatial")