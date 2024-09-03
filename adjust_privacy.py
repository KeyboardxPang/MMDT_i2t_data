import pandas as pd
import json
from glob import glob
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

# sort the image_paths by the file name
# the file name is in the form of "index_9_4.jpg", so we need to extract the index
# firstly sort the image by the first number in the file name
# then sort the image by the second number in the file name

def extract_number(file_path):
    first_number = int(file_path.split("/")[-1].split("_")[1])
    second_number = int(file_path.split("/")[-1].split("_")[2].split(".")[0])
    return first_number * 10 + second_number

df_street = pd.read_csv("/Users/yuanlingzhi/Desktop/change_data/data/image-to-text/privacy/label_sum.csv")

data_list = []

folder_path = "/Users/yuanlingzhi/Desktop/change_data/data/image-to-text/privacy/Pri-Street-View/Dataset-1-single_image_text"
task = "street_view"
type_street_view = folder_path.split("/")[-1].split("-")[-1]
print(type_street_view)
image_paths = glob(folder_path + "/*.jpg")
image_paths.sort(key=extract_number)

for image_path in image_paths:
    data = {}
    data["task"] = task
    data["image"] = image_path
    data["id"] = image_path.split("/")[-1].split(".")[0]
    image_index = int(data["id"].split("_")[1])
    data["type_street_view"] = type_street_view
    data["country"] = df_street[df_street["image_index"] == image_index]["country"].values[0]
    data["state_province"] = df_street[df_street["image_index"] == image_index]["state_province"].values[0]
    data["city"] = df_street[df_street["image_index"] == image_index]["city"].values[0]
    data["latitude"] = df_street[df_street["image_index"] == image_index]["latitude"].values[0]
    data["longitude"] = df_street[df_street["image_index"] == image_index]["longitude"].values[0]
    data["zipcode"] = df_street[df_street["image_index"] == image_index]["zipcode"].values[0]
    data["ethnicity"] = ""
    data["label_selfies"] = ""
    data["type_selfies"] = ""
    data_list.append(data)

folder_path = "/Users/yuanlingzhi/Desktop/change_data/data/image-to-text/privacy/Pri-Street-View/Dataset-2-single_image_no_text"
task = "street_view"
type_street_view = folder_path.split("/")[-1].split("-")[-1]
print(type_street_view)
image_paths = glob(folder_path + "/*.jpg")
image_paths.sort(key=extract_number)

for image_path in image_paths:
    data = {}
    data["task"] = task
    data["image"] = image_path
    data["id"] = image_path.split("/")[-1].split(".")[0]
    image_index = int(data["id"].split("_")[1])
    data["type_street_view"] = type_street_view
    data["country"] = df_street[df_street["image_index"] == image_index]["country"].values[0]
    data["state_province"] = df_street[df_street["image_index"] == image_index]["state_province"].values[0]
    data["city"] = df_street[df_street["image_index"] == image_index]["city"].values[0]
    data["latitude"] = df_street[df_street["image_index"] == image_index]["latitude"].values[0]
    data["longitude"] = df_street[df_street["image_index"] == image_index]["longitude"].values[0]
    data["zipcode"] = df_street[df_street["image_index"] == image_index]["zipcode"].values[0]
    data["ethnicity"] = ""
    data["label_selfies"] = ""
    data["type_selfies"] = ""
    data_list.append(data)

folder_path = "/Users/yuanlingzhi/Desktop/change_data/data/image-to-text/privacy/Pri-Street-View/Dataset-3-group_image_no_text"
task = "street_view"
type_street_view = folder_path.split("/")[-1].split("-")[-1]
print(type_street_view)
image_paths = glob(folder_path + "/*.jpg")
image_paths.sort(key=extract_number)

for image_path in image_paths:
    data = {}
    data["task"] = task
    data["image"] = image_path
    data["id"] = image_path.split("/")[-1].split(".")[0]
    image_index = int(data["id"].split("_")[1])
    data["type_street_view"] = type_street_view
    data["country"] = df_street[df_street["image_index"] == image_index]["country"].values[0]
    data["state_province"] = df_street[df_street["image_index"] == image_index]["state_province"].values[0]
    data["city"] = df_street[df_street["image_index"] == image_index]["city"].values[0]
    data["latitude"] = df_street[df_street["image_index"] == image_index]["latitude"].values[0]
    data["longitude"] = df_street[df_street["image_index"] == image_index]["longitude"].values[0]
    data["zipcode"] = df_street[df_street["image_index"] == image_index]["zipcode"].values[0]
    data["ethnicity"] = ""
    data["label_selfies"] = ""
    data["type_selfies"] = ""
    data_list.append(data)

folder_path = "/Users/yuanlingzhi/Desktop/change_data/data/image-to-text/privacy/Pri-Street-View/Dataset-4-group_image_text"
task = "street_view"
type_street_view = folder_path.split("/")[-1].split("-")[-1]
print(type_street_view)
image_paths = glob(folder_path + "/*.jpg")
image_paths.sort(key=extract_number)

for image_path in image_paths:
    data = {}
    data["task"] = task
    data["image"] = image_path
    data["id"] = image_path.split("/")[-1].split(".")[0]
    image_index = int(data["id"].split("_")[1])
    data["type_street_view"] = type_street_view
    data["country"] = df_street[df_street["image_index"] == image_index]["country"].values[0]
    data["state_province"] = df_street[df_street["image_index"] == image_index]["state_province"].values[0]
    data["city"] = df_street[df_street["image_index"] == image_index]["city"].values[0]
    data["latitude"] = df_street[df_street["image_index"] == image_index]["latitude"].values[0]
    data["longitude"] = df_street[df_street["image_index"] == image_index]["longitude"].values[0]
    data["zipcode"] = df_street[df_street["image_index"] == image_index]["zipcode"].values[0]
    data["ethnicity"] = ""
    data["label_selfies"] = ""
    data["type_selfies"] = ""
    data_list.append(data)

# save the data to a .json file
file_path = "/Users/yuanlingzhi/Desktop/change_data/test_data/privacy/street_view.json"
with open(file_path, 'w') as f:
    json.dump(data_list, f, indent=4)

dataset = Dataset.from_json(file_path)
dataset = dataset.cast_column("image", Image())

# push the dataset to huggingface repository
dataset.push_to_hub("YuanXiaopang/test_mmdt", config_name="privacy", split="street_view")
