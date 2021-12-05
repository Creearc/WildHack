import sys
import os
import time
import numpy as np
import cv2
import imutils


def resize(img, w=None, h=None):
  sh, sw = img.shape[:2]
  if not (w is None):
    k = sw / w
  elif not (h is None):
    k = sh / h  
  w, h = int(sw // k), int(sh // k)
  return cv2.resize(img, (w, h))

def combine(base_img_rgb, next_img_rgb):
  base_img = cv2.GaussianBlur(cv2.cvtColor(base_img_rgb, cv2.COLOR_BGR2GRAY), (5,5), 0)

  #use SIFT feature detector
  detector = cv2.xfeatures2d.SIFT_create()

  # Parameters for nearest-neighbor matching
  FLANN_INDEX_KDTREE = 1
  flann_params = dict(algorithm = FLANN_INDEX_KDTREE, 
      trees = 5)
  matcher = cv2.FlannBasedMatcher(flann_params, {})

  # Read in the next image...
  next_img = cv2.GaussianBlur(cv2.cvtColor(next_img_rgb,cv2.COLOR_BGR2GRAY), (5,5), 0)

  # Find points in the next frame
  base_features, base_descs = detector.detectAndCompute(base_img, None)
  next_features, next_descs = detector.detectAndCompute(next_img, None)

  matches = matcher.knnMatch(next_descs, trainDescriptors=base_descs, k=2)

  kp1, kp2 = [], []

  for match in matches:
    kp1.append(base_features[match.trainIdx])
    kp2.append(next_features[match.queryIdx])

  p1 = np.array([k.pt for k in kp1])
  p2 = np.array([k.pt for k in kp2])

  H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)

  # how successful the transformation was
  print('%d / %d  inliers/matched' % (np.sum(status), len(status)))

  base_img_warp = cv2.warpPerspective(base_img_rgb, move_h, (img_w, img_h))
  next_img_warp = cv2.warpPerspective(closestImage['rgb'], mod_inv_h, (img_w, img_h))

  # Put the base image on an enlarged palette
  enlarged_base_img = np.zeros((img_h, img_w, 3), np.uint8)

  # Create masked composite
  (ret,data_map) = cv2.threshold(cv2.cvtColor(next_img_warp, cv2.COLOR_BGR2GRAY), 
      0, 255, cv2.THRESH_BINARY)

  enlarged_base_img = cv2.add(enlarged_base_img, base_img_warp, 
      mask=np.bitwise_not(data_map), 
      dtype=cv2.CV_8U)

  # Now add the warped image
  final_img = cv2.add(enlarged_base_img, next_img_warp, 
      dtype=cv2.CV_8U)

  out = resize(final_img.copy(), h=1080)
  cv2.imshow('', out)
  cv2.waitKey(0)

  



path = 'dataset/p2'

images = []

for f in os.listdir(path):
  print('{}/{}'.format(path, f))
  img = cv2.imread('{}/{}'.format(path, f))
  images.append(img)

  if len(images) > 1:
    result = combine(result, images[-1])
  else:
    result = images[0].copy()

cv2.destroyAllWindows() 


