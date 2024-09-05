import pandas as pd
import json
from glob import glob
from datasets import Dataset, Image
import os

# each data has 6 keys
# common keys:
# 1. question
# 2. image
# 3. id
# 4. task

# cooccurence has:
# 5. label
# 6. target

# misleading and ocr have:
# 7. keyword

# counterfactual, distraction and natural have:
# 8. answer
# 9. bbox
# 10. natural_question
# 11. natural_answer

# # misleading split
# json_root_path = "/Users/yuanlingzhi/Desktop/change_data/data/image-to-text/hallucination/misleading"
# task_list = ["action", "attribute", "count", "identification", "spatial"]
# all_data_list = []

# for task in task_list:
#     file_path = os.path.join(json_root_path, f"{task}/dataset.json")
#     image_folder_path = os.path.join(json_root_path, f"{task}/images")

#     # load the json file
#     with open(file_path, 'r') as f:
#         data_list = json.load(f)
    
#     for data in data_list:
#         data["task"] = task
#         data["target"] = ""
#         data["label"] = ""
#         data["image"] = os.path.join(image_folder_path, data["image_path"].split("/")[-1])
#         data.pop("image_path")
#         data["answer"] = ""
#         data["bbox"] = ""
#         data["natural_question"] = ""
#         data["natural_answer"] = ""
    
#     all_data_list.extend(data_list)

# # save the data to a .json file
# file_path = "/Users/yuanlingzhi/Desktop/change_data/MMDT_i2t_data/test_data/hallucination/misleading.json"
# with open(file_path, 'w') as f:
#     json.dump(all_data_list, f, indent=4)

# dataset = Dataset.from_json(file_path)
# dataset = dataset.cast_column("image", Image())

# # push the dataset to huggingface repository
# dataset.push_to_hub("YuanXiaopang/test_mmdt", config_name="hallucination", split="misleading")

# ocr split
json_root_path = "/Users/yuanlingzhi/Desktop/change_data/data/image-to-text/hallucination/ocr"
task_list = ["contradictory", "cooccur", "doc", "scene"]
all_data_list = []

for task in task_list:
    file_path = os.path.join(json_root_path, f"{task}/dataset.json")
    image_folder_path = os.path.join(json_root_path, f"{task}/images")

    # load the json file
    with open(file_path, 'r') as f:
        data_list = json.load(f)
    
    for data in data_list:
        data["task"] = task
        data["target"] = ""
        data["label"] = ""
        data["image"] = os.path.join(image_folder_path, data["image_path"].split("/")[-1])
        data.pop("image_path")
        data["answer"] = ""
        data["bbox"] = ""
        data["natural_question"] = ""
        data["natural_answer"] = ""
    
    # delete the data that id>124
    data_list = [data for data in data_list if data["id"] <= 124]
    all_data_list.extend(data_list)

# save the data to a .json file
file_path = "/Users/yuanlingzhi/Desktop/change_data/MMDT_i2t_data/test_data/hallucination/ocr.json"
with open(file_path, 'w') as f:
    json.dump(all_data_list, f, indent=4)

dataset = Dataset.from_json(file_path)
dataset = dataset.cast_column("image", Image())

# push the dataset to huggingface repository
dataset.push_to_hub("YuanXiaopang/test_mmdt", config_name="hallucination", split="ocr")

