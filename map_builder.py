import sys
import os
import time
import numpy as np
import cv2
import imutils

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

def matchKeyPointsKNN(featuresA, featuresB, ratio):
  rawMatches = bf.knnMatch(featuresA, featuresB, 2)
  matches = []

  for m,n in rawMatches:
    if m.distance < n.distance * ratio:
        matches.append(m)
  return matches

def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
    ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
    ptsB = np.float32([kpsB[m.trainIdx] for m in matches])
  
    (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
        reprojThresh)

    return (H, status)

def combine_arr(base, img, coords):
  if coords is None:
    coords = (0, orig.shape[1], 0, orig.shape[0])
  
  result = np.zeros((base.shape[0] + img.shape[0] * 2,
                     base.shape[1] + img.shape[1] * 2, 3),
                    np.uint8)

  orig = base[coords[0] : coords[1], coords[2] : coords[3]].copy()
  orig_gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
  orig_kps, orig_features = cv2.ORB_create().detectAndCompute(orig_gray, None)
  orig_kps = np.float32([kp.pt for kp in orig_kps])

  img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  img_kps, img_features = cv2.ORB_create().detectAndCompute(img_gray, None)
  img_kps = np.float32([kp.pt for kp in img_kps])
  
  matches = matchKeyPointsKNN(img_features, orig_features, ratio=0.75)
  if len(matches) < 4:
    print('Low count of matches')
    return base, coords
  M = getHomography(img_kps, orig_kps, img_features, orig_features, matches, reprojThresh=4)
  (H, status) = M

  width = orig.shape[1] + img.shape[1]
  height = orig.shape[0] + img.shape[0]

  tmp = cv2.warpPerspective(img, H, (width, height))
  gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
  mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

  cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)

  c = max(cnts, key=cv2.contourArea)
  (x1, y1, w1, h1) = cv2.boundingRect(c)

  result[img.shape[0] : img.shape[0] + base.shape[0],
         img.shape[1] : img.shape[1] + base.shape[1]] = base

  tmp_0 = result[coords[0] + img.shape[0] : coords[0] + img.shape[0] + height,
         coords[2] + img.shape[1]: coords[2] + img.shape[1]+ width]

  tmp = cv2.bitwise_and(tmp, tmp, mask=mask)
  tmp_0 = cv2.bitwise_and(tmp_0, tmp_0, mask=cv2.bitwise_not(mask))
      
  result[coords[0] + img.shape[0] : coords[0] + img.shape[0] + height,
         coords[2] + img.shape[1]  : coords[2] + img.shape[1]  + width] = cv2.add(tmp_0, tmp)

  h, w = tmp.shape[:2]
  k = h / 1080
  w, h = int(w // k), int(h // k)
  print(w, h)
  cv2.rectangle(tmp, (x1, y1),(x1+w1, y1+h1), (0,255,0), 3)
  tmp = cv2.resize(tmp, (w, h))

  cv2.imshow('', tmp)
  cv2.waitKey(0)

  gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

  cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)

  c = max(cnts, key=cv2.contourArea)
  (x, y, w, h) = cv2.boundingRect(c)
  result = result[y : y + h, x : x + w, :]
  
  return result, (y1, y1+h1,
                  x1, x1+w1)



path = 'dataset/2020'
  
arr = []
coords = None
old_img = None
result = None

for f in os.listdir(path):
  print('{}/{}'.format(path, f))
  img = cv2.imread('{}/{}'.format(path, f))
  img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

  kps, features = cv2.ORB_create().detectAndCompute(img_gray, None)
  kps = np.float32([kp.pt for kp in kps])

  arr.append((kps, features, img))

  if old_img is None:
    old_img = img.copy()
    result = img.copy()
    coords = (0, img.shape[0], 0, img.shape[1])
    continue
  
  result, coords_new = combine_arr(result, img, coords)
  #coords = np.add(coords, coords_new)
  coords = coords_new
  print(result.shape)
  print(coords)

  old_img = img.copy()
  
  out = result.copy()
  cv2.rectangle(out, (coords[2], coords[0]),(coords[3], coords[1]), (0,255,0), 3)
  h, w = out.shape[:2]
  k = h / 1080
  w, h = int(w // k), int(h // k)
  print(w, h)
  out = cv2.resize(out, (w, h))
  cv2.imshow('', out)
  key = cv2.waitKey(0)
  if key != 32:
    break
  
cv2.destroyAllWindows()  

