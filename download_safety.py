from datasets import load_dataset

ds = load_dataset("danielz01/mmdt")

# upload the dataset to the hub
ds.push_to_hub("YuanXiaopang/test_mmdt", config_name="safety")