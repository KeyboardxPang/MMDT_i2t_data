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

image_folder_path = "/Users/yuanlingzhi/Desktop/change_data/data/image-to-text/hallucination/cooccurrence/images"
task_list = ["action", "attribute", "count", "identification", "spatial"]

# cooccurrence_high_cooc split
data_list = []
json_root_path = "/Users/yuanlingzhi/Desktop/change_data/data/image-to-text/hallucination/cooccurrence/high_cooc"

for task in task_list:
    file_path = os.path.join(json_root_path, f"{task}.json")

    # each line is a json object
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            data["task"] = task
            data["target"] = str(data["target"])
            data["image"] = os.path.join(image_folder_path, data["image_path"])
            # remove the key "image_path"
            data.pop("image_path")
            # change the key "idx" to "id"
            data["id"] = data.pop("idx")
            data["keyword"] = ""
            data["question"] = data.pop("prompt")
            data["answer"] = ""
            data["bbox"] = ""
            data["natural_question"] = ""
            data["natural_answer"] = ""
            data_list.append(data)

# save the data to a .json file
file_path = "/Users/yuanlingzhi/Desktop/change_data/MMDT_i2t_data/test_data/hallucination/cooccurrence_high_cooc.json"
with open(file_path, 'w') as f:
    json.dump(data_list, f, indent=4)

dataset = Dataset.from_json(file_path)
dataset = dataset.cast_column("image", Image())

# push the dataset to huggingface repository
dataset.push_to_hub("YuanXiaopang/test_mmdt", config_name="hallucination", split="cooccurrence_high_cooc")

# cooccurrence_historical_bias
data_list = []
json_root_path = "/Users/yuanlingzhi/Desktop/change_data/data/image-to-text/hallucination/cooccurrence/historical_bias"

for task in task_list:
    file_path = os.path.join(json_root_path, f"{task}.json")

    # each line is a json object
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            data["task"] = task
            data["target"] = str(data["target"])
            data["image"] = os.path.join(image_folder_path, data["image_path"])
            # remove the key "image_path"
            data.pop("image_path")
            # change the key "idx" to "id"
            data["id"] = data.pop("idx")
            data["keyword"] = ""
            data["question"] = data.pop("prompt")
            data["answer"] = ""
            data["bbox"] = ""
            data["natural_question"] = ""
            data["natural_answer"] = ""
            data_list.append(data)

# save the data to a .json file
file_path = "/Users/yuanlingzhi/Desktop/change_data/MMDT_i2t_data/test_data/hallucination/cooccurrence_historical_bias.json"
with open(file_path, 'w') as f:
    json.dump(data_list, f, indent=4)

dataset = Dataset.from_json(file_path)
dataset = dataset.cast_column("image", Image())

# push the dataset to huggingface repository
dataset.push_to_hub("YuanXiaopang/test_mmdt", config_name="hallucination", split="cooccurrence_historical_bias")

# cooccurrence_low_cooc
data_list = []
json_root_path = "/Users/yuanlingzhi/Desktop/change_data/data/image-to-text/hallucination/cooccurrence/low_cooc"

for task in task_list:
    file_path = os.path.join(json_root_path, f"{task}.json")

    # each line is a json object
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            data["task"] = task
            data["target"] = str(data["target"])
            data["image"] = os.path.join(image_folder_path, data["image_path"])
            # remove the key "image_path"
            data.pop("image_path")
            # change the key "idx" to "id"
            data["id"] = data.pop("idx")
            data["keyword"] = ""
            data["question"] = data.pop("prompt")
            data["answer"] = ""
            data["bbox"] = ""
            data["natural_question"] = ""
            data["natural_answer"] = ""
            data_list.append(data)

# save the data to a .json file
file_path = "/Users/yuanlingzhi/Desktop/change_data/MMDT_i2t_data/test_data/hallucination/cooccurrence_low_cooc.json"
with open(file_path, 'w') as f:
    json.dump(data_list, f, indent=4)

dataset = Dataset.from_json(file_path)
dataset = dataset.cast_column("image", Image())

# push the dataset to huggingface repository
dataset.push_to_hub("YuanXiaopang/test_mmdt", config_name="hallucination", split="cooccurrence_low_cooc")
