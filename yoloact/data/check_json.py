import json

with open(r'D:/hackathon/WildHack/yolact/examples_data/cig_butts/waste/val/coco_annotations.json', 'r') as f:
    json_data = json.load(f)
    
print(json.dumps(json_data, indent=2))