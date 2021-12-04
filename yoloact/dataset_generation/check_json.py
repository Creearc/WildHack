import json

with open(r'/home/alexandr/wildhack/WildHack/yoloact/dataset_generation/train/output/instances_waste_train2021.json', 'r') as f:
    json_data = json.load(f)
    
print(json.dumps(json_data, indent=2))