# WildHack

Команда для запуска программы сбора картинок в одну. Зависит от мощности компьютера. Нам удалось объединять до 150 картинок.
Результат сохраняется в папку results\odm_orthophoto в файл odm_orthophoto.png

```
python odm_test.py
```



Команда для создания файла baseline_solution.csv и масок с положениями объектов по классам:

```
python eval.py --trained_model=./weights/yolact_resnet50_wildhack_waste_308_88000.pth --config=yolact_resnet50_wildhack_waste_config --score_threshold=0.6 --top_k=25 --images=../examples_data/cig_butts/cig_butts/real_test:output_images_big
```



Команда для создания nikita_solution.csv - файл с результатми приближенными к тем, что использовались при разметке организаторами хакатона. Нужно изменить пути к папке с масками и к папке с картинками.

```
python dil.py
```

