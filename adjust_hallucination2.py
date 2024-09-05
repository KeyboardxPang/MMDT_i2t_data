import pandas as pd
import json
from glob import glob
from natsort import natsorted
from datasets import Dataset, Image
import os
import pandas as pd

# each data has 6 keys
# common keys:
# 1. question
# 2. image
# 3. id
# 4. task

# cooccurence has:
# 5. label
# 6. target
# 7. keyword

# counterfactual has:
# 8. answer
# 9. bbox
# 10. natural_question
# 11. natural_answer

root_path = "/Users/yuanlingzhi/Desktop/change_data/data/image-to-text/hallucination/counterfactual"
task_list = ["action", "attribute", "count", "identification", "spatial"]
data_list = [] 
for task in task_list:
    image_folder_path = os.path.join(root_path, task)
    csv_path = os.path.join(image_folder_path, f"{task}.csv")
    if not os.path.exists(csv_path):
        continue
    csv_file = pd.read_csv(csv_path)
    data = {}
    for idx, row in csv_file.iterrows():
        data["task"] = task
        data["target"] = ""
        data["keyword"] = ""
        data["label"] = ""
        data["id"] = row["img_id"]
        data["image"] = os.path.join(image_folder_path, f"{row['img_id']}.png")
        data["question"] = row["question"]
        data["answer"] = row["answer"]
        if "bbox" in row:
            data["bbox"] = row["bbox"]
        else:
            data["bbox"] = ""
        if "natural_question" in row:
            data["natural_question"] = row["natural_question"]
        else:
            data["natural_question"] = ""
        if "natural_answer" in row:
            data["natural_answer"] = row["natural_answer"]
        else:
            data["natural_answer"] = ""
        data_list.append(data)

# save the data to a .json file
file_path = "/Users/yuanlingzhi/Desktop/change_data/MMDT_i2t_data/test_data/hallucination/counterfactual.json"
with open(file_path, 'w') as f:
    json.dump(data_list, f, indent=4)

dataset = Dataset.from_json(file_path)
dataset = dataset.cast_column("image", Image())

# push the dataset to huggingface repository
dataset.push_to_hub("YuanXiaopang/test_mmdt", config_name="hallucination", split="counterfactual")

root_path = "/Users/yuanlingzhi/Desktop/change_data/data/image-to-text/hallucination/distraction"
task_list = ["action", "attribute", "count", "identification", "spatial"]
data_list = [] 
for task in task_list:
    image_folder_path = os.path.join(root_path, task)
    csv_file = pd.read_csv(os.path.join(image_folder_path, f"{task}.csv"))
    data = {}
    for idx, row in csv_file.iterrows():
        data["task"] = task
        data["target"] = ""
        data["keyword"] = ""
        data["label"] = ""
        data["id"] = row["img_id"]
        data["image"] = os.path.join(image_folder_path, f"{row['img_id']}.png")
        data["question"] = row["question"]
        data["answer"] = row["answer"]
        data["answer"] = row["answer"]
        if "bbox" in row:
            data["bbox"] = row["bbox"]
        else:
            data["bbox"] = ""
        if "natural_question" in row:
            data["natural_question"] = row["natural_question"]
        else:
            data["natural_question"] = ""
        if "natural_answer" in row:
            data["natural_answer"] = row["natural_answer"]
        else:
            data["natural_answer"] = ""
        data_list.append(data)

# save the data to a .json file
file_path = "/Users/yuanlingzhi/Desktop/change_data/MMDT_i2t_data/test_data/hallucination/distraction.json"
with open(file_path, 'w') as f:
    json.dump(data_list, f, indent=4)

dataset = Dataset.from_json(file_path)
dataset = dataset.cast_column("image", Image())

# push the dataset to huggingface repository
dataset.push_to_hub("YuanXiaopang/test_mmdt", config_name="hallucination", split="distraction")

root_path = "/Users/yuanlingzhi/Desktop/change_data/data/image-to-text/hallucination/natural"
task_list = ["action", "attribute", "count", "identification", "spatial"]
data_list = [] 
for task in task_list:
    image_folder_path = os.path.join(root_path, task)
    csv_file = pd.read_csv(os.path.join(image_folder_path, f"{task}.csv"))
    data = {}
    for idx, row in csv_file.iterrows():
        data["task"] = task
        data["target"] = ""
        data["keyword"] = ""
        data["label"] = ""
        data["id"] = row["img_id"]
        data["image"] = os.path.join(image_folder_path, f"{row['img_id']}.png")
        data["question"] = row["question"]
        data["answer"] = row["answer"]
        data["answer"] = row["answer"]
        if "bbox" in row:
            data["bbox"] = row["bbox"]
        else:
            data["bbox"] = ""
        if "natural_question" in row:
            data["natural_question"] = row["natural_question"]
        else:
            data["natural_question"] = ""
        if "natural_answer" in row:
            data["natural_answer"] = row["natural_answer"]
        else:
            data["natural_answer"] = ""
        data_list.append(data)

# save the data to a .json file
file_path = "/Users/yuanlingzhi/Desktop/change_data/MMDT_i2t_data/test_data/hallucination/natural.json"
with open(file_path, 'w') as f:
    json.dump(data_list, f, indent=4)

dataset = Dataset.from_json(file_path)
dataset = dataset.cast_column("image", Image())

# push the dataset to huggingface repository
dataset.push_to_hub("YuanXiaopang/test_mmdt", config_name="hallucination", split="natural")