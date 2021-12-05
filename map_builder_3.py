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

def compute_kps_and_features(img):
  img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  img_kps, img_features = cv2.ORB_create().detectAndCompute(img_gray, None) ###
  img_kps = np.float32([kp.pt for kp in img_kps])

  return img_kps, img_features


def build_map(arr1, arr2):
  img_1, coords_1 = arr1
  img_2, coords_2 = arr2

  img_1_kps, img_1_features = compute_kps_and_features(img_1)
  img_2_kps, img_2_features = compute_kps_and_features(img_2)  

  result = np.zeros((img_1.shape[0] + img_2.shape[0] * 10,
                     img_1.shape[1] + img_2.shape[1] * 10, 3),
                    np.uint8)

  print('Result -> ', result.shape[:2])

  result[img_2.shape[0] : img_2.shape[0] + img_1.shape[0],
         img_2.shape[1] : img_2.shape[1] + img_1.shape[1]] = img_1


  tmp = img_2.copy()
  height, width = tmp.shape[:2]

  dx, dy = int(ptd * (coords_1[1] - coords_2[1])), int(ptd * (coords_1[0] - coords_2[0]))

  img_coords = (img_2.shape[0] + img_1.shape[0] // 2 - dy,
                img_2.shape[0] + img_1.shape[0] // 2 + height - dy,
                
                img_2.shape[1] + img_1.shape[1] // 2 - dx,
                img_2.shape[1] + img_1.shape[1] // 2 + width - dx)
  
  print('img_coords -> ', img_coords)
  
  tmp_0 = result[img_coords[0] : img_coords[1],
                 img_coords[2] : img_coords[3]]

  print('tmp_0 -> ', tmp_0.shape[:2])
  print('tmp -> ', tmp.shape[:2])

  result[img_coords[0] : img_coords[1],
         img_coords[2] : img_coords[3]] = np.where(tmp_0 == 0, tmp, tmp_0)
  
  return result


path = 'dataset/p2'

ptd = 0.1

dd = 10 ** 8
  
arr = []
coords = None
old_img = None
result = None

for f in os.listdir(path):
  print('{}/{}'.format(path, f))
  img = cv2.imread('{}/{}'.format(path, f))
  with open('{}/{}'.format(path, f), "rb") as file:
    image = Image(file)
    coords = (dd * dms_coordinates_to_dd_coordinates(image.gps_latitude, image.gps_latitude_ref),
              dd * dms_coordinates_to_dd_coordinates(image.gps_longitude, image.gps_longitude_ref))

    print(coords[0], coords[1])

    arr.append((img, coords))

    if len(arr) == 2:
      break

  
result = build_map(arr[0], arr[1])  
    
out = result.copy()
out = resize(out, h=1080)
cv2.imshow('', out)
cv2.waitKey(0)
  
cv2.destroyAllWindows()  





















