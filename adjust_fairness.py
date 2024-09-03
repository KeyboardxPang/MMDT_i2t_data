import pandas as pd
import json
from glob import glob
from datasets import Dataset, Image

def extract_number(file_path):
    return int(file_path.split("/")[-1].split(".")[0])

# # read the .json file as a list of dictionary
# file_path = "/Users/yuanlingzhi/Desktop/change_data/test_data/fairness/group_activity.json"
# with open(file_path, 'r') as f:
#     data = json.load(f)

# img_folder_path = "/Users/yuanlingzhi/Desktop/change_data/test_data/fairness/group_activity"
# img_paths = glob(img_folder_path + "/*.png")

# # sort the img_paths by the file name
# img_paths.sort(key=extract_number)

# # change the "img_path" key to each dictionary
# for i in range(len(data)):
#     data[i]["image"] = img_paths[i]
#     # delete the "img_path" key
#     del data[i]["img_path"]

# # save the data to the original .json file
# with open(file_path, 'w') as f:
#     json.dump(data, f, indent=4)

# dataset = Dataset.from_json(file_path)
# dataset = dataset.cast_column("image", Image())

# # push the dataset to huggingface repository
# dataset.push_to_hub("YuanXiaopang/test_mmdt", config_name="fairness", split="group_activity")


# # read the .json file as a list of dictionary
# file_path = "/Users/yuanlingzhi/Desktop/change_data/test_data/fairness/group_education.json"
# with open(file_path, 'r') as f:
#     data = json.load(f)

# img_folder_path = "/Users/yuanlingzhi/Desktop/change_data/test_data/fairness/group_education"
# img_paths = glob(img_folder_path + "/*.png")

# # sort the img_paths by the file name
# img_paths.sort(key=extract_number)

# # change the "img_path" key to each dictionary
# for i in range(len(data)):
#     data[i]["image"] = img_paths[i]
#     # delete the "img_path" key
#     del data[i]["img_path"]

# # save the data to the original .json file
# with open(file_path, 'w') as f:
#     json.dump(data, f, indent=4)

# dataset = Dataset.from_json(file_path)
# dataset = dataset.cast_column("image", Image())

# # push the dataset to huggingface repository
# dataset.push_to_hub("YuanXiaopang/test_mmdt", config_name="fairness", split="group_education")


# # read the .json file as a list of dictionary
# file_path = "/Users/yuanlingzhi/Desktop/change_data/test_data/fairness/group_occupation.json"
# with open(file_path, 'r') as f:
#     data = json.load(f)

# img_folder_path = "/Users/yuanlingzhi/Desktop/change_data/test_data/fairness/group_occupation"
# img_paths = glob(img_folder_path + "/*.png")

# # sort the img_paths by the file name
# img_paths.sort(key=extract_number)

# # change the "img_path" key to each dictionary
# for i in range(len(data)):
#     data[i]["image"] = img_paths[i]
#     # delete the "img_path" key
#     del data[i]["img_path"]

# # save the data to the original .json file
# with open(file_path, 'w') as f:
#     json.dump(data, f, indent=4)

# dataset = Dataset.from_json(file_path)
# dataset = dataset.cast_column("image", Image())

# # push the dataset to huggingface repository
# dataset.push_to_hub("YuanXiaopang/test_mmdt", config_name="fairness", split="group_occupation")


img_folder_path = "/Users/yuanlingzhi/Desktop/change_data/test_data/fairness/individual"
# glob all the files in the img_folder_path
img_paths = glob(img_folder_path + "/*")

# sort the img_paths by the file name
img_paths.sort(key=extract_number)

# create a json file
data = []
for i, img_path in enumerate(img_paths):
    data.append({
        "q_gender": "",
        "q_race": "",
        "q_age": "",
        "image": img_path
    })

# save the data to a .json file
file_path = "/Users/yuanlingzhi/Desktop/change_data/test_data/fairness/individual.json"
with open(file_path, 'w') as f:
    json.dump(data, f, indent=4)

dataset = Dataset.from_json(file_path)
dataset = dataset.cast_column("image", Image())

# push the dataset to huggingface repository
dataset.push_to_hub("YuanXiaopang/test_mmdt", config_name="fairness", split="individual")