import sys
import os
import time
import numpy as np
import cv2
import imutils

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

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

def matchKeyPointsKNN(featuresA, featuresB, ratio):
  rawMatches = bf.knnMatch(featuresA, featuresB, 2) ###
  matches = []

  for m, n in rawMatches:
    if m.distance < n.distance * ratio:
        matches.append(m)
  return matches

def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
    ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
    ptsB = np.float32([kpsB[m.trainIdx] for m in matches])
  
    (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
        reprojThresh)

    return (H, status)

def match_check(featuresA, featuresB, ratio=0.75):
  rawMatches = bf.knnMatch(featuresA, featuresB, 2)
  matches = 0

  for m, n in rawMatches:
    if m.distance < n.distance * ratio:
        matches += 1
  return matches

def compute_kps_and_features(img):
  img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  img_kps, img_features = cv2.ORB_create().detectAndCompute(img_gray, None) ###
  img_kps = np.float32([kp.pt for kp in img_kps])

  return img_kps, img_features

def combine_arr(base, img, coords):
  if coords is None:
    coords = (0, orig.shape[1], 0, orig.shape[0])
  
  result = np.zeros((base.shape[0] + img.shape[0] * 2,
                     base.shape[1] + img.shape[1] * 2, 3),
                    np.uint8)

  tmp_res = np.zeros(((coords[1] - coords[0]) * 2,
                     (coords[3] - coords[2]) * 2, 3),
                    np.uint8)

  orig = base[coords[0] : coords[1], coords[2] : coords[3]].copy()
  width = orig.shape[1] + img.shape[1]
  height = orig.shape[0] + img.shape[0]

  part = base[coords[0] : coords[1],
              coords[2] : coords[3]].copy()

  print('Part -> ', part.shape[:2])
  print('Coords -> ', coords[1] - coords[0], coords[3] - coords[2], coords)

  #debug_show(resize(part.copy(), h=1080), name='part')

  h, w = tmp_res.shape[:2]
  dy, dx = coords[1] - coords[0], coords[3] - coords[2]
  tmp_res[h // 2 - dy // 2 : h // 2 - dy // 2 + dy,
          w // 2 - dx // 2 : w // 2 - dx // 2 + dx] = part
  
  orig = tmp_res.copy()
  
  orig_kps, orig_features = compute_kps_and_features(orig)
  
  matches = matchKeyPointsKNN(img_features, orig_features, ratio=0.95)
  print('Matches -> ', len(matches))
  if len(matches) < 10:
    print('Low count of matches')
    return base, coords
  M = getHomography(img_kps, orig_kps, img_features, orig_features, matches, reprojThresh=4)
  (H, status) = M

  tmp = cv2.warpPerspective(img, H, (width, height))
  gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
  mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

  cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)

  c = max(cnts, key=cv2.contourArea)
  (x1, y1, w1, h1) = cv2.boundingRect(c)

  dx, dy = coords[3] // 2, coords[1] // 2

  result[img.shape[0] : img.shape[0] + base.shape[0],
         img.shape[1] : img.shape[1] + base.shape[1]] = base

  img_coords = (coords[0] + img.shape[0] - dy, coords[0] + img.shape[0] + height - dy,
                coords[2] + img.shape[1] - dx, coords[2] + img.shape[1] + width - dx)

  tmp_0 = result[img_coords[0] : img_coords[1],
                 img_coords[2] : img_coords[3]]
     
  result[img_coords[0] : img_coords[1],
         img_coords[2] : img_coords[3]] = np.where(tmp_0 == 0, tmp, tmp_0)
  
  if not True:
    cv2.rectangle(tmp, (x1, y1),(x1+w1, y1+h1), (0,255,0), 10)
    tmp = resize(tmp, h=1080)

    cv2.imshow('', tmp)
    cv2.waitKey(0)

  gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

  cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)

  c = max(cnts, key=cv2.contourArea)
  (x, y, w, h) = cv2.boundingRect(c)
  result = result[y : y + h, x : x + w, :]

  print('Here -> ', y, x, img_coords[0] , img_coords[2])

  dy, dx = np.clip(y - img_coords[0] - y1, 0, result.shape[0]), np.clip(x - img_coords[2] - x1, 0, result.shape[1])
  dy, dx = img_coords[0] + y1 - y, img_coords[2] + x1 - x

  k = 0
  return result, (dy + int(h1 * k), dy + h1 - int(h1 * k),
                  dx + int(w1 * k), dx + w1 - int(w1 * k))



path = 'dataset/p2'
  
arr = []
coords = None
old_img = None
result = None

for f in os.listdir(path):
  print('{}/{}'.format(path, f))
  img = cv2.imread('{}/{}'.format(path, f))
  #img = resize(img, h=720)
  img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

  kps, features = cv2.ORB_create().detectAndCompute(img_gray, None)
  kps = np.float32([kp.pt for kp in kps])

  if old_img is None:
    old_img = img.copy()
    result = img.copy()
    coords = (0, img.shape[0], 0, img.shape[1])
    continue

  img_kps, img_features = compute_kps_and_features(img)
  
  mx = 0
  c = (0, img.shape[0], 0, img.shape[1])
  for element in arr:
    similarity = match_check(element[1], img_features)
    print(similarity)
    if mx == 0 or similarity > mx:
      mx = similarity
      c = element[3]     
  
  result, coords = combine_arr(result, img, c)
  print(result.shape)
  print(coords)

  arr.append((img_kps, img_features, img, coords))

  if len(arr) > 5:
    arr.pop(0)

  old_img = img.copy()
  
  out = result.copy()
  cv2.rectangle(out, (coords[2], coords[0]),(coords[3], coords[1]), (0,255,0), 10)
  
  out = resize(out, h=1080)
  cv2.imshow('', out)
  key = cv2.waitKey(1)
  if key != 32 and key != -1:
    break
  
cv2.destroyAllWindows()  

