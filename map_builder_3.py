import sys
import os
import time
import numpy as np
import cv2
import imutils
from exif import Image

def resize(img, w=None, h=None):
  sh, sw = img.shape[:2]
  if not (w is None):
    k = sw / w
  elif not (h is None):
    k = sh / h  
  w, h = int(sw // k), int(sh // k)
  return cv2.resize(img, (w, h))

def debug_show(img, name=''):
  tmp = resize(img, h=1080)
  cv2.imshow(name, img)
  cv2.waitKey(0)

def dms_coordinates_to_dd_coordinates(coordinates, coordinates_ref):
    decimal_degrees = coordinates[0] + \
                      coordinates[1] / 60 + \
                      coordinates[2] / 3600   
    if coordinates_ref == "S" or coordinates_ref == "W":
        decimal_degrees = -decimal_degrees    
    return decimal_degrees


def build_map():
  return result

path = 'dataset/p2'
  
arr = []
coords = None
old_img = None
result = None

for f in os.listdir(path):
  print('{}/{}'.format(path, f))
  img = cv2.imread('{}/{}'.format(path, f))
  with open('{}/{}'.format(path, f), "rb") as file:
    image = Image(file)
    coords = (dms_coordinates_to_dd_coordinates(image.gps_latitude, image.gps_latitude_ref),
              dms_coordinates_to_dd_coordinates(image.gps_longitude, image.gps_longitude_ref))

    print(coords[0], coords[1])

  

  
    

out = resize(arr[0][2], h=1080)
cv2.imshow('', out)
cv2.waitKey(0)
  
cv2.destroyAllWindows()  





















