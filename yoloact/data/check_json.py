import json

with open(r'/home/alexandr/wildhack/WildHack/yoloact/data/waste/val/coco_annotations.json', 'r') as f:
    json_data = json.load(f)
    
print(json.dumps(json_data, indent=2))