# WildHack

Команда для создания файла baseline_solution.csv и масок с положениями объектов по классам:

'''
python eval.py --trained_model=./weights/yolact_resnet50_wildhack_waste_308_88000.pth --config=yolact_resnet50_wildhack_waste_config --score_threshold=0.6 --top_k=25 --images=../examples_data/cig_butts/cig_butts/real_test:output_images_big
'''
