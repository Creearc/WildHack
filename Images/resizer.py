import sys
import os
import numpy as np
import cv2

paths = ['metal', 'net', 'plastic', 'wood']

out_path = 'imgs_out'

k = 5

for path in paths:
  for f in os.listdir(path):
    print('{}/{}'.format(path, f))
    img = cv2.imread('{}/{}'.format(path, f), cv2.IMREAD_UNCHANGED)
    if int(f.split('_')[-1].split('.')[0]) > 100:
      h, w = img.shape[:2]
      img = cv2.resize(img, (w // k, h // k))

    cv2.imwrite('{}/{}'.format(out_path, f), img)
      
  


